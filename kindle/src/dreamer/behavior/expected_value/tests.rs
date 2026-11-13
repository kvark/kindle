use super::*;
use crate::dreamer::runtime::build_session;
use meganeura::Mode;
use std::sync::Arc;

#[test]
fn expected_value_config_preserves_legacy_and_rejects_actor_combinations() {
    let mut config = DreamerConfig::tiny(3);
    let legacy = serde_json::to_value(&config).unwrap();
    assert!(legacy.get("actor_expected_value").is_none());
    assert_eq!(
        serde_json::from_value::<DreamerConfig>(legacy).unwrap(),
        config
    );
    config.actor_expected_value = true;
    assert!(config.check().is_ok());
    assert_eq!(
        serde_json::from_value::<DreamerConfig>(serde_json::to_value(&config).unwrap()).unwrap(),
        config
    );
    for variant in 0..5 {
        let mut bad = config.clone();
        match variant {
            0 => bad.actor_dynamics_scale = Some(0.5),
            1 => bad.replay_actor_scale = Some(1.0),
            2 => bad.actor_critic_gradient = true,
            3 => bad.actor_residual = true,
            _ => bad.loss_scales.policy = 0.0,
        }
        assert!(bad.check().is_err());
    }
}

#[test]
fn expected_value_graph_keeps_parameter_owners_and_compiles() {
    let mut config = DreamerConfig::tiny(3);
    config.batch_size = 1;
    config.batch_length = 2;
    config.world_backprop_length = 2;
    config.imagination_length = 3;
    let plain = super::super::build_training_graph(&config, 6, 1);
    config.actor_expected_value = true;
    let combined = super::super::build_training_graph(&config, 6, 1);
    let parameters = |graph: &Graph| {
        graph
            .nodes()
            .iter()
            .filter_map(|n| {
                if let Op::Parameter { name } = &n.op {
                    Some((name.clone(), n.ty.clone()))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(parameters(&plain), parameters(&combined));
    assert_eq!(
        combined.outputs().len(),
        super::super::expected_value_actor_output(&config) + 1
    );
    let (plan, _) = meganeura::compile_training_graph(&combined);
    assert!(
        plan.input_buffers
            .iter()
            .any(|(n, _)| n == "expected_latent_uniforms")
    );
    assert!(!plan.input_buffers.iter().any(|(n, _)| n == "action_target"));
    let mut graph = Graph::new();
    let model = super::super::BehaviorModel::new(&mut graph, &config);
    let weight = graph.input("weights", &[6, 1]);
    let loss = loss(&mut graph, &config, &model.actor.base, &model.value, weight);
    graph.set_outputs(vec![loss]);
    let (plan, _) = meganeura::compile_training_graph(&graph);
    let mut actor_count = 0;
    for (name, buffer) in &plan.param_buffers {
        let differentiated = plan.param_grad_pairs.iter().any(|(p, _)| p == buffer);
        assert_eq!(
            differentiated,
            name.starts_with("behavior.actor."),
            "{name}"
        );
        actor_count += usize::from(differentiated);
        assert!(!name.starts_with("world."));
    }
    assert_eq!(actor_count, 11);
    assert!(
        plan.input_buffers
            .iter()
            .any(|(n, _)| n.starts_with("expected_parameter_world."))
    );
    for actions in [3, 18] {
        config.action_count = actions;
        let graph = build_values_graph(&config);
        assert_eq!(graph.node(graph.outputs()[1]).ty.shape, [2, actions]);
        meganeura::compile_plan(
            &graph,
            Mode::Inference,
            Default::default(),
            &Default::default(),
        );
    }
}

#[test]
fn expected_value_branch_layout_repeats_each_state_and_draw() {
    let mut config = DreamerConfig::tiny(3);
    config.batch_size = 1;
    config.batch_length = 2;
    let mut graph = Graph::new();
    let (_, deter, stoch, action, uniforms) = branch_inputs(&mut graph, &config);
    let size = config.network();
    assert_eq!(graph.node(deter).ty.shape, [6, size.deter]);
    assert_eq!(graph.node(stoch).ty.shape, [6 * size.stoch, size.classes]);
    assert_eq!(graph.node(uniforms).ty.shape, graph.node(stoch).ty.shape);
    assert_eq!(graph.node(action).ty.shape, [6, 3]);
    if let Op::Constant { ref data } = graph.node(action).op {
        assert_eq!(
            data,
            &[
                1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 1.
            ]
        );
    } else {
        panic!("action branches must be explicit one-hot constants");
    }
}

#[test]
#[ignore = "requires separately guarded expected-value objective qualification"]
fn expected_value_native_objective_and_actor_derivatives() {
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    assert_eq!(
        gpu.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    let logits = [0.4f64, -0.6, 0.2, 0.1, -0.2, 0.3];
    let q = [2.0f64, -1.0, 5.0, -2.0, 3.0, 1.0];
    let weights = [0.9f64, 0.4];
    let mut graph = Graph::new();
    let actor = graph.parameter("actor", &[2, 3]);
    let values = graph.parameter("critic_values", &[2, 3]);
    let weight = graph.parameter("weights", &[2, 1]);
    let scale = graph.input("scale", &[1]);
    let loss = objective(&mut graph, actor, values, weight, scale, 0.01);
    graph.set_outputs(vec![loss]);
    let mut session = build_session(&graph, &gpu, Mode::Training, false);
    session.set_parameter("actor", &logits.map(|x| x as f32));
    session.set_parameter("critic_values", &q.map(|x| x as f32));
    session.set_parameter("weights", &weights.map(|x| x as f32));
    session.set_input("scale", &[2.0]);
    assert!(session.has_param_grad("actor"));
    assert!(!session.has_param_grad("critic_values") && !session.has_param_grad("weights"));
    let before = session.read_params(&["critic_values", "weights"]);
    session.set_adam(1e-4, 0.9, 0.999, 1e-8);
    session.step();
    session.wait();
    let mut expected = 0.0;
    let mut derivatives = [0.0; 6];
    for row in 0..2 {
        let values = &q[row * 3..row * 3 + 3];
        let l = &logits[row * 3..row * 3 + 3];
        let max = l.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut p = l.iter().map(|x| (x - max).exp()).collect::<Vec<_>>();
        let sum: f64 = p.iter().sum();
        p.iter_mut().for_each(|x| *x /= sum);
        let center: f64 = values.iter().sum::<f64>() / 3.0;
        let baseline: f64 = p.iter().zip(values).map(|(p, q)| p * q).sum();
        for action in 0..3 {
            expected -=
                weights[row] * (0.99 * p[action] + 0.01 / 3.0) * (values[action] - center) / 4.0;
            derivatives[row * 3 + action] =
                -weights[row] * 0.99 * p[action] * (values[action] - baseline) / 4.0;
        }
    }
    let mut actual = [0.0];
    let mut gradients = [0.0; 6];
    session.read_output_by_index(0, &mut actual);
    session.read_param_grad("actor", &mut gradients);
    assert!((f64::from(actual[0]) - expected).abs() < 2e-5);
    for (actual, expected) in gradients.into_iter().zip(derivatives) {
        assert!(expected.abs() > 1e-4);
        assert!((f64::from(actual) - expected).abs() < 2e-5);
    }
    assert_eq!(session.read_params(&["critic_values", "weights"]), before);
    session.set_parameter("weights", &[0.0; 2]);
    session.step();
    session.wait();
    session.read_output_by_index(0, &mut actual);
    session.read_param_grad("actor", &mut gradients);
    assert_eq!(actual[0], 0.0);
    assert!(gradients.iter().all(|&x| x == 0.0));
}
