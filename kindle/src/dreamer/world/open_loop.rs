//! Supervised prior rollouts from one posterior start per replay sequence.

use super::*;

pub(super) struct Losses {
    /// Scaled auxiliary total, unscaled CDP/reward/continuation means.
    pub outputs: [NodeId; 4],
    #[cfg(test)]
    pub states: Vec<NodeId>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build(
    graph: &mut Graph,
    config: &DreamerConfig,
    model: &WorldModel,
    mut deter: NodeId,
    mut stoch: NodeId,
    actions: &[NodeId],
    encodings: &[NodeId],
    center: Option<NodeId>,
) -> Losses {
    let rollout = config.open_loop_supervision.as_ref().unwrap();
    let rows = config.batch_size;
    let size = config.network();
    let mut prediction_losses = Vec::new();
    let mut reward_losses = Vec::new();
    let mut continuation_losses = Vec::new();
    #[cfg(test)]
    let mut states = Vec::new();
    for time in 1..=rollout.horizon {
        // Only the start state is posterior. No future encoding enters dynamics.
        deter = model
            .dynamics
            .core
            .forward(graph, deter, stoch, actions[time], rows);
        let logits = model.dynamics.prior.forward(graph, deter, rows);
        let uniforms = graph.input(
            &format!("rollout_uniforms_{time}"),
            &[rows * size.stoch, size.classes],
        );
        let sample = gumbel_sample(
            graph,
            logits,
            uniforms,
            rows * size.stoch,
            size.classes,
            config.unimix,
        );
        let sample = graph.stop_gradient(sample);
        let probabilities = mixed_probabilities(
            graph,
            logits,
            rows * size.stoch,
            size.classes,
            config.unimix,
        );
        stoch = straight_through_sample(graph, sample, probabilities);
        let weight = graph.input(&format!("rollout_weight_{time}"), &[rows, 1]);
        // The first deterministic prediction already has the same inputs as
        // the ordinary one-step CDP term. Add new CDP targets only from step 2.
        if time >= 2 {
            let prediction = model
                .future_predictor
                .as_ref()
                .unwrap()
                .forward(graph, deter, rows);
            let prediction = graph.reshape(prediction, &[rows, config.prediction_dim()]);
            let target = graph.stop_gradient(encodings[time]);
            let (prediction, target) = if let Some(center) = center {
                (
                    graph.bias_add(prediction, center),
                    graph.bias_add(target, center),
                )
            } else {
                (prediction, target)
            };
            let distance = cdp::cosine_distance(graph, prediction, target);
            let weighted = graph.mul(distance, weight);
            prediction_losses.push(graph.mean_all(weighted));
        }
        let state = feature(graph, deter, stoch, rows, config);
        #[cfg(test)]
        states.push(state);
        let reward = model.heads.reward.forward(graph, state);
        let continuation = model.heads.continuation.forward(graph, state);
        let reward_target = graph.input(
            &format!("rollout_reward_target_{time}"),
            &[rows, config.value_bins],
        );
        let continuation_target =
            graph.input(&format!("rollout_continuation_target_{time}"), &[rows, 1]);
        reward_losses.push(weighted_cross_entropy(graph, reward, reward_target, weight));
        continuation_losses.push(weighted_binary_cross_entropy(
            graph,
            continuation,
            continuation_target,
            weight,
        ));
    }
    let prediction = sum(graph, &prediction_losses);
    let prediction = graph.scale(prediction, 1.0 / (rollout.horizon - 1) as f32);
    let reward = sum(graph, &reward_losses);
    let reward = graph.scale(reward, 1.0 / rollout.horizon as f32);
    let continuation = sum(graph, &continuation_losses);
    let continuation = graph.scale(continuation, 1.0 / rollout.horizon as f32);
    let losses = [
        graph.scale(prediction, config.loss_scales.future_prediction),
        graph.scale(reward, config.loss_scales.reward),
        graph.scale(continuation, config.loss_scales.continuation),
    ];
    let total = sum(graph, &losses);
    let total = graph.scale(total, rollout.scale);
    Losses {
        outputs: [total, prediction, reward, continuation],
        #[cfg(test)]
        states,
    }
}

fn weighted_binary_cross_entropy(
    graph: &mut Graph,
    logits: NodeId,
    target: NodeId,
    weight: NodeId,
) -> NodeId {
    let positive = graph.softplus(logits, 1.0);
    let product = graph.mul(logits, target);
    let per_row = graph.sub(positive, product);
    let weighted = graph.mul(per_row, weight);
    graph.mean_all(weighted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use meganeura::graph::Op;
    use std::collections::HashSet;

    fn config() -> DreamerConfig {
        let mut config = DreamerConfig::tiny(3);
        config.observation_kind = ObservationKind::Rgb32;
        config.loss_scales.reconstruction = 0.0;
        config.loss_scales.future_prediction = 500.0;
        config.batch_length = 4;
        config.world_backprop_length = 4;
        config.cdp_centered = true;
        config.open_loop_supervision = Some(super::super::super::OpenLoopSupervision {
            horizon: 3,
            scale: 1.0,
        });
        config
    }

    fn fixture(config: &DreamerConfig) -> (Graph, Vec<NodeId>) {
        let mut graph = Graph::new();
        let model = WorldModel::new(&mut graph, config);
        let size = config.network();
        let deter = graph.parameter("start_deter", &[config.batch_size, size.deter]);
        let stoch = graph.parameter(
            "start_stoch",
            &[config.batch_size, size.stoch * size.classes],
        );
        let encodings = (0..4)
            .map(|t| {
                graph.parameter(
                    &format!("encoding_{t}"),
                    &[config.batch_size, config.prediction_dim()],
                )
            })
            .collect::<Vec<_>>();
        let actions = (0..4)
            .map(|t| {
                graph.input(
                    &format!("action_{t}"),
                    &[config.batch_size, config.action_count],
                )
            })
            .collect::<Vec<_>>();
        let targets = stack_time(
            &mut graph,
            &encodings,
            config.batch_size,
            config.prediction_dim(),
        );
        let center = cdp::negative_target_mean(&mut graph, targets);
        let losses = build(
            &mut graph,
            config,
            &model,
            deter,
            stoch,
            &actions,
            &encodings,
            Some(center),
        );
        graph.set_outputs(
            losses
                .outputs
                .into_iter()
                .chain(losses.states.iter().copied())
                .collect(),
        );
        (graph, losses.states)
    }

    fn ancestors(graph: &Graph, root: NodeId) -> HashSet<NodeId> {
        let mut found = HashSet::new();
        let mut pending = vec![root];
        while let Some(id) = pending.pop() {
            if found.insert(id) {
                pending.extend(&graph.node(id).inputs);
            }
        }
        found
    }

    #[test]
    fn prior_rollout_has_only_past_actions_and_no_future_observations() {
        let config = config();
        let (graph, states) = fixture(&config);
        for (step, state) in states.iter().enumerate() {
            let dependencies = ancestors(&graph, *state);
            for node in graph.nodes() {
                match &node.op {
                    Op::Parameter { name } if name.starts_with("encoding_") => {
                        assert!(!dependencies.contains(&node.id))
                    }
                    Op::Input { name } if name.starts_with("action_") => {
                        let time = name[7..].parse::<usize>().unwrap();
                        assert_eq!(
                            dependencies.contains(&node.id),
                            time > 0 && time <= step + 1
                        );
                    }
                    _ => (),
                }
            }
        }
        let backward = meganeura::autodiff::differentiate(&graph);
        let gradients = &backward.outputs()[graph.outputs().len()..];
        for (node, &gradient) in graph
            .nodes()
            .iter()
            .filter(|node| matches!(node.op, Op::Parameter { .. }))
            .zip(gradients)
        {
            let Op::Parameter { name } = &node.op else {
                unreachable!()
            };
            if name.starts_with("encoding_") || name.starts_with("behavior.") {
                assert_ne!(
                    backward.node(gradient).ty.shape,
                    node.ty.shape,
                    "detached {name}"
                );
            } else if name.starts_with("world.dynamics.")
                || name.starts_with("world.future_predictor.")
                || name.starts_with("world.reward.")
                || name.starts_with("world.continuation.")
                || name.starts_with("start_")
            {
                assert_eq!(
                    backward.node(gradient).ty.shape,
                    node.ty.shape,
                    "connected {name}"
                );
            }
        }
    }

    #[test]
    fn optional_rollout_preserves_parameters_and_inference_and_compiles() {
        let enabled = config();
        assert!(enabled.check().is_ok());
        let mut disabled = enabled.clone();
        disabled.open_loop_supervision = None;
        let graph = build_training_graph(&enabled, 4);
        let control = build_training_graph(&disabled, 4);
        let parameters = |graph: &Graph| {
            graph
                .nodes()
                .iter()
                .filter_map(|node| match &node.op {
                    Op::Parameter { name } => Some((name.clone(), node.ty.shape.clone())),
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(parameters(&graph), parameters(&control));
        assert_eq!(
            format!("{:?}", build_observe_graph(&enabled, 2).nodes()),
            format!("{:?}", build_observe_graph(&disabled, 2).nodes())
        );
        let (plan, _) = meganeura::compile_training_graph(&graph);
        assert!(!plan.dispatches.is_empty());
        let staged = build_staged_training_graph(&enabled);
        assert_eq!(staged.outputs().len(), LOSS_OPEN_LOOP_CONTINUATION + 3);
        let _ = meganeura::autodiff::differentiate(&staged);
    }

    #[test]
    #[ignore = "requires separately guarded GPU; prior rollout gradients, causal invariance and zero masks"]
    fn rollout_gradients_and_alignment() {
        use crate::dreamer::runtime::{build_session, initialize_d3};
        let config = config();
        let (graph, _) = fixture(&config);
        let gpu = std::sync::Arc::new(crate::init_gpu_context().unwrap());
        assert_eq!(
            gpu.device_information().device_name,
            std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
        );
        assert!(!gpu.device_information().is_software_emulated);
        let mut session = build_session(&graph, &gpu, meganeura::Mode::Training, false);
        initialize_d3(&mut session, &graph, 19);
        let rows = config.batch_size;
        let size = config.network();
        let bins = crate::dreamer::distributions::TwoHotBins::new(config.value_bins);
        for time in 1..=3 {
            let mut action = vec![0.0; rows * config.action_count];
            for row in 0..rows {
                action[row * config.action_count + time % config.action_count] = 1.0;
            }
            let mut target = vec![0.0; rows * config.value_bins];
            for row in 0..rows {
                bins.encode(
                    time as f32,
                    &mut target[row * config.value_bins..(row + 1) * config.value_bins],
                );
            }
            session.set_input(&format!("action_{time}"), &action);
            session.set_input(
                &format!("rollout_uniforms_{time}"),
                &vec![0.37; rows * size.stoch * size.classes],
            );
            session.set_input(&format!("rollout_weight_{time}"), &vec![1.0; rows]);
            session.set_input(&format!("rollout_reward_target_{time}"), &target);
            session.set_input(
                &format!("rollout_continuation_target_{time}"),
                &vec![if time == 3 { 0.0 } else { 0.99 }; rows],
            );
        }
        session.step();
        session.wait();
        let read = |session: &meganeura::Session, index, size| {
            let mut values = vec![0.0; size];
            session.read_output_by_index(index, &mut values);
            values
        };
        let baseline = read(&session, 0, 1)[0];
        assert!(baseline.is_finite() && baseline > 0.0);
        let states = (4..7)
            .map(|index| read(&session, index, rows * config.feature_dim()))
            .collect::<Vec<_>>();
        let names = session
            .param_names()
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        for name in &names {
            if name.starts_with("encoding_") || name.starts_with("behavior.") {
                assert!(!session.has_param_grad(name), "detached {name}");
            }
        }
        for prefix in [
            "world.dynamics.core.",
            "world.dynamics.prior",
            "world.future_predictor.",
            "world.reward.",
            "world.continuation.",
            "start_deter",
            "start_stoch",
        ] {
            let mut norm = 0.0_f64;
            for name in names.iter().filter(|name| name.starts_with(prefix)) {
                assert!(session.has_param_grad(name), "connected {name}");
                let mut gradient = vec![0.0; session.param_size(name).unwrap()];
                session.read_param_grad(name, &mut gradient);
                assert!(gradient.iter().all(|x| x.is_finite()));
                norm += gradient.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>();
            }
            assert!(norm > 1e-14, "zero {prefix}");
        }
        // A smooth head parameter has a true finite-difference derivative;
        // categorical straight-through paths are checked by routing above.
        let name = "world.reward.out.bias";
        let mut gradient = vec![0.0; config.value_bins];
        session.read_param_grad(name, &mut gradient);
        let index = (0..gradient.len())
            .max_by(|&a, &b| gradient[a].abs().total_cmp(&gradient[b].abs()))
            .unwrap();
        let mut head = vec![0.0; config.value_bins];
        session.read_param(name, &mut head);
        let original = head[index];
        head[index] = original + 0.01;
        session.set_parameter(name, &head);
        session.step();
        session.wait();
        let positive = read(&session, 2, 1)[0];
        head[index] = original - 0.01;
        session.set_parameter(name, &head);
        session.step();
        session.wait();
        let negative = read(&session, 2, 1)[0];
        assert!(((positive - negative) / 0.02 - gradient[index]).abs() < 1e-3);
        head[index] = original;
        session.set_parameter(name, &head);
        session.set_parameter("encoding_2", &vec![0.25; rows * config.prediction_dim()]);
        session.step();
        session.wait();
        assert!((read(&session, 0, 1)[0] - baseline).abs() > 1e-4);
        for (index, state) in states.iter().enumerate() {
            assert_eq!(read(&session, index + 4, state.len()), *state);
        }
        session.set_input("action_2", &vec![0.0; rows * config.action_count]);
        session.step();
        session.wait();
        assert_eq!(read(&session, 4, states[0].len()), states[0]);
        assert_ne!(read(&session, 5, states[1].len()), states[1]);
        assert_ne!(read(&session, 6, states[2].len()), states[2]);
        for time in 1..=3 {
            session.set_input(&format!("rollout_weight_{time}"), &vec![0.0; rows]);
        }
        session.step();
        session.wait();
        assert_eq!(read(&session, 0, 1), [0.0]);
        for name in &names {
            if session.has_param_grad(name) {
                let mut gradient = vec![0.0; session.param_size(name).unwrap()];
                session.read_param_grad(name, &mut gradient);
                assert!(gradient.iter().all(|&x| x == 0.0), "masked {name}");
            }
        }
    }

    #[test]
    #[ignore = "requires separately guarded GPU; masked logits BCE values and derivatives against F64"]
    fn masked_continuation_matches_f64() {
        use crate::dreamer::runtime::build_session;
        let logits = [-80.0_f32, -5.0, 0.0, 5.0, 80.0, 2.0];
        let targets = [1.0_f32, 0.0, 0.7, 1.0, 0.0, 0.0];
        let weights = [1.0_f32, 1.0, 0.5, 1.0, 1.0, 0.0];
        let mut graph = Graph::new();
        let x = graph.parameter("logits", &[6, 1]);
        let target = graph.constant(targets.to_vec(), &[6, 1]);
        let weight = graph.constant(weights.to_vec(), &[6, 1]);
        let loss = weighted_binary_cross_entropy(&mut graph, x, target, weight);
        graph.set_outputs(vec![loss]);
        let gpu = std::sync::Arc::new(crate::init_gpu_context().unwrap());
        assert_eq!(
            gpu.device_information().device_name,
            std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
        );
        assert!(!gpu.device_information().is_software_emulated);
        let mut session = build_session(&graph, &gpu, meganeura::Mode::Training, false);
        session.set_parameter("logits", &logits);
        session.step();
        session.wait();
        let mut actual = [0.0];
        let mut gradients = [0.0; 6];
        session.read_output_by_index(0, &mut actual);
        session.read_param_grad("logits", &mut gradients);
        let mut expected = 0.0;
        for i in 0..6 {
            let x = f64::from(logits[i]);
            let y = f64::from(targets[i]);
            let w = f64::from(weights[i]);
            expected += (x.max(0.0) + (-x.abs()).exp().ln_1p() - x * y) * w / 6.0;
            let gradient = (1.0 / (1.0 + (-x).exp()) - y) * w / 6.0;
            assert!(
                (f64::from(gradients[i]) - gradient).abs() < 2e-6,
                "{i}: {} vs {gradient}",
                gradients[i]
            );
        }
        assert!((f64::from(actual[0]) - expected).abs() < 2e-5);
    }
}
