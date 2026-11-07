use super::*;
use crate::dreamer::{ExpertImitation, ObservationKind, config::ExpertImitationDecay};
use meganeura::{
    Graph,
    graph::{NodeId, Op},
};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn config() -> DreamerConfig {
    let mut config = DreamerConfig::tiny(3);
    config.observation_kind = ObservationKind::Rgb32;
    config.loss_scales.reconstruction = 0.0;
    config.loss_scales.future_prediction = 500.0;
    config.cdp_centered = true;
    config.actor_residual = true;
    config.expert_imitation = Some(ExpertImitation {
        scale: 2.0,
        action_weights: vec![1.0; 3],
        world_gradient: true,
        decay: None,
    });
    config.validate();
    config
}

fn parameters(graph: &Graph) -> BTreeMap<String, Vec<usize>> {
    graph
        .nodes()
        .iter()
        .filter_map(|node| match &node.op {
            Op::Parameter { name } => Some((name.clone(), node.ty.shape.clone())),
            _ => None,
        })
        .collect()
}

fn differentiable_parameters(graph: &Graph, output: NodeId) -> BTreeSet<String> {
    let mut pending = vec![output];
    let mut seen = BTreeSet::new();
    let mut parameters = BTreeSet::new();
    while let Some(id) = pending.pop() {
        if !seen.insert(id) {
            continue;
        }
        let node = graph.node(id);
        match &node.op {
            Op::Parameter { name } => {
                parameters.insert(name.clone());
            }
            Op::StopGradient => {}
            _ => pending.extend(&node.inputs),
        }
    }
    parameters
}

#[test]
fn residual_config_defaults_restore_and_scope() {
    let legacy = DreamerConfig::tiny(3);
    let json = serde_json::to_value(&legacy).unwrap();
    assert!(json.get("actor_residual").is_none());
    assert_eq!(
        serde_json::from_value::<DreamerConfig>(json).unwrap(),
        legacy
    );
    let enabled = config();
    let json = serde_json::to_value(&enabled).unwrap();
    assert_eq!(json["actor_residual"], true);
    assert_eq!(
        serde_json::from_value::<DreamerConfig>(json).unwrap(),
        enabled
    );
    let mutations: &[fn(&mut DreamerConfig)] = &[
        |c| c.observation_kind = ObservationKind::Features,
        |c| c.cdp_centered = false,
        |c| c.expert_imitation = None,
        |c| c.expert_imitation.as_mut().unwrap().world_gradient = false,
        |c| c.expert_imitation.as_mut().unwrap().scale = 0.0,
        |c| {
            c.expert_imitation.as_mut().unwrap().decay = Some(ExpertImitationDecay {
                start_update: 1,
                end_update: 2,
                actor_only: true,
            })
        },
        |c| c.loss_scales.policy = 0.0,
        |c| c.actor_critic_gradient = true,
        |c| c.actor_dynamics_scale = Some(0.5),
        |c| c.replay_actor_scale = Some(1.0),
        |c| c.actor_learning_starts = 1,
    ];
    for change in mutations {
        let mut changed = enabled.clone();
        change(&mut changed);
        assert!(changed.check().unwrap_err().contains("residual actor"));
    }
}

#[test]
fn residual_shapes_and_loss_routes_match_all_owners() {
    let config = config();
    let graph = behavior::build_training_graph(&config, 2, 2);
    let residual = BTreeSet::from([
        "behavior.actor.residual.weight".to_owned(),
        "behavior.actor.residual.bias".to_owned(),
    ]);
    for output in [
        behavior::LOSS_POLICY,
        behavior::POLICY_ENTROPY,
        behavior::WEIGHTED_POLICY_ENTROPY,
    ] {
        assert_eq!(
            differentiable_parameters(&graph, graph.outputs()[output]),
            residual
        );
    }
    let expert = differentiable_parameters(&graph, graph.outputs()[behavior::LOSS_EXPERT]);
    assert_eq!(expert.len(), 11);
    assert!(
        expert
            .iter()
            .all(|name| name.starts_with("behavior.actor.") && !residual.contains(name))
    );
    let mut plain = config.clone();
    plain.actor_residual = false;
    for (enabled, disabled) in [
        (graph, behavior::build_training_graph(&plain, 2, 2)),
        (
            behavior::build_actor_inference_graph(&config, 2),
            behavior::build_actor_inference_graph(&plain, 2),
        ),
        (
            world::build_imagination_graph(&config),
            world::build_imagination_graph(&plain),
        ),
    ] {
        let mut actual = parameters(&enabled);
        assert_eq!(
            actual.remove("behavior.actor.residual.weight"),
            Some(vec![config.feature_dim(), 3])
        );
        assert_eq!(actual.remove("behavior.actor.residual.bias"), Some(vec![3]));
        assert_eq!(actual, parameters(&disabled));
    }
    let enabled = world::build_training_graph(&config, config.world_backprop_length);
    let disabled = world::build_training_graph(&plain, plain.world_backprop_length);
    assert_eq!(
        format!("{:?}", enabled.nodes()),
        format!("{:?}", disabled.nodes())
    );
    assert_eq!(enabled.outputs(), disabled.outputs());
    assert!(
        !parameters(&enabled)
            .keys()
            .any(|name| name.contains(".residual."))
    );
}

fn reference(
    logits: &[f32],
    features: &[f32],
    targets: &[f32],
    weights: &[f32],
    config: &DreamerConfig,
) -> (f64, Vec<f64>, Vec<f64>) {
    let actions = config.action_count;
    let features_count = config.feature_dim();
    let rows = weights.len();
    let mut loss = 0.0;
    let mut dw = vec![0.0; features_count * actions];
    let mut db = vec![0.0; actions];
    for row in 0..rows {
        let logits = &logits[row * actions..(row + 1) * actions];
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let exp = logits
            .iter()
            .map(|&x| (f64::from(x) - max).exp())
            .collect::<Vec<_>>();
        let total: f64 = exp.iter().sum();
        let soft = exp.iter().map(|x| x / total).collect::<Vec<_>>();
        let p = soft
            .iter()
            .map(|x| {
                (1.0 - f64::from(config.actor_unimix)) * x
                    + f64::from(config.actor_unimix) / actions as f64
            })
            .collect::<Vec<_>>();
        let mut derivative = vec![0.0; actions];
        for action in 0..actions {
            let target = f64::from(targets[row * actions + action]);
            let entropy = f64::from(config.actor_entropy * weights[row]);
            loss += (-target * p[action].ln() + entropy * p[action] * p[action].ln()) / rows as f64;
            derivative[action] = -target / p[action] + entropy * (p[action].ln() + 1.0);
        }
        let average: f64 = soft.iter().zip(&derivative).map(|(s, d)| s * d).sum();
        for action in 0..actions {
            let dz = (1.0 - f64::from(config.actor_unimix))
                * soft[action]
                * (derivative[action] - average)
                / rows as f64;
            db[action] += dz;
            for feature in 0..features_count {
                dw[feature * actions + action] +=
                    f64::from(features[row * features_count + feature]) * dz;
            }
        }
    }
    (loss, dw, db)
}

#[test]
#[ignore = "requires separately declared guarded residual actor qualification"]
fn residual_native_logits_derivatives_and_gradient_routing() {
    let config = config();
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    let device = crate::gpu_device_info(gpu.device_information());
    assert_eq!(
        device.device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    assert!(!device.is_software_emulated);
    let mut graph = behavior::build_training_graph(&config, 2, 2);
    let outputs = graph.outputs().to_vec();
    let policy_gate = graph.input("check_policy", &[1]);
    let expert_gate = graph.input("check_expert", &[1]);
    let policy = graph.mul(outputs[behavior::LOSS_POLICY], policy_gate);
    let expert = graph.mul(outputs[behavior::LOSS_EXPERT], expert_gate);
    let selected = graph.add(policy, expert);
    graph.set_outputs(std::iter::once(selected).chain(outputs).collect());
    let mut session = super::super::runtime::build_training_session(&graph, &gpu, &config);
    initialize_d3(&mut session, &graph, config.seed);
    session.clear_optimizer();
    session.disable_grad_clip();
    let mut plain = config.clone();
    plain.actor_residual = false;
    let base_graph = behavior::build_actor_inference_graph(&plain, 2);
    let combined_graph = behavior::build_actor_inference_graph(&config, 2);
    let mut base = build_session(&base_graph, &gpu, Mode::Inference, false);
    let mut combined = build_session(&combined_graph, &gpu, Mode::Inference, false);
    for target in [&mut base, &mut combined] {
        sync_matching(&session, target, "behavior.actor.");
    }
    let features = (0..2 * config.feature_dim())
        .map(|i| (i % 13) as f32 * 0.03 - 0.18)
        .collect::<Vec<_>>();
    base.set_input("feature", &features);
    combined.set_input("feature", &features);
    base.step();
    base.wait();
    combined.step();
    combined.wait();
    let base_logits = base.read_output(6);
    assert_eq!(
        base_logits,
        combined.read_output(6),
        "zero correction must preserve logits"
    );
    for name in [
        "behavior.actor.residual.weight",
        "behavior.actor.residual.bias",
    ] {
        assert!(session.read_params(&[name])[0].iter().all(|&x| x == 0.0));
    }
    let weights = (0..config.feature_dim() * 3)
        .map(|i| (i % 7) as f32 * 0.001 - 0.003)
        .collect::<Vec<_>>();
    let biases = [0.01, -0.02, 0.03];
    session.set_parameter("behavior.actor.residual.weight", &weights);
    session.set_parameter("behavior.actor.residual.bias", &biases);
    sync_matching(&session, &mut combined, "behavior.actor.");
    combined.step();
    combined.wait();
    let logits = combined.read_output(6);
    let mut expected_logits = vec![0.0; 6];
    for row in 0..2 {
        for action in 0..3 {
            let correction: f64 = (0..config.feature_dim())
                .map(|f| {
                    f64::from(features[row * config.feature_dim() + f])
                        * f64::from(weights[f * 3 + action])
                })
                .sum();
            let expected =
                f64::from(base_logits[row * 3 + action]) + correction + f64::from(biases[action]);
            expected_logits[row * 3 + action] = expected as f32;
            assert!((f64::from(logits[row * 3 + action]) - expected).abs() < 2e-5);
        }
    }
    let inputs = graph
        .nodes()
        .iter()
        .filter_map(|node| match &node.op {
            Op::Input { name } => Some((name.clone(), node.ty.shape.iter().product())),
            _ => None,
        })
        .collect::<Vec<(String, usize)>>();
    for (name, size) in inputs {
        session.set_input(&name, &vec![0.0; size]);
    }
    session.set_input("imagined_feature", &features);
    session.set_input("replay_feature", &features);
    let targets = [1.0, 0.0, 0.0, 0.0, -0.4, 0.0];
    let trajectory_weights = [0.9, 0.5];
    session.set_input("action_target", &targets);
    session.set_input("imagined_weight", &trajectory_weights);
    session.set_input("expert_action_target", &[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
    let (loss, dw, db) = reference(
        &expected_logits,
        &features,
        &targets,
        &trajectory_weights,
        &config,
    );
    let names = session
        .param_names()
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let borrowed = names.iter().map(String::as_str).collect::<Vec<_>>();
    let before = session.read_params(&borrowed);
    for (policy, expert) in [(1.0, 0.0), (0.0, 1.0)] {
        session.set_input("check_policy", &[policy]);
        session.set_input("check_expert", &[expert]);
        session.zero_grad();
        session.step();
        session.wait();
        if policy == 1.0 {
            assert!((f64::from(session.read_loss()) - loss).abs() < 2e-5);
        }
        let mut nonzero = false;
        for name in &names {
            if !session.has_param_grad(name) {
                continue;
            }
            let mut gradient = vec![0.0; session.param_size(name).unwrap()];
            session.read_param_grad(name, &mut gradient);
            assert!(gradient.iter().all(|x| x.is_finite()));
            let residual = name.starts_with("behavior.actor.residual.");
            let actor = name.starts_with("behavior.actor.");
            if !actor || (policy == 1.0 && !residual) || (expert == 1.0 && residual) {
                assert!(
                    gradient.iter().all(|&x| x == 0.0),
                    "forbidden derivative {name}"
                );
            } else {
                nonzero |= gradient.iter().any(|&x| x != 0.0);
            }
            if policy == 1.0 && residual {
                let expected = if name.ends_with(".weight") { &dw } else { &db };
                for (&actual, &expected) in gradient.iter().zip(expected) {
                    assert!((f64::from(actual) - expected).abs() < 2e-5 * (1.0 + expected.abs()));
                }
            }
        }
        assert!(nonzero);
        assert_eq!(session.read_params(&borrowed), before);
        assert_eq!(session.adam_step_count(), 0);
    }
    eprintln!(
        "Residual actor affine logits, F64 policy derivatives, loss routing and frozen parameters pass; zero optimizer/emulator steps"
    );
}
