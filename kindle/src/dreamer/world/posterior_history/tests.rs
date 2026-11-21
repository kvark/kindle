use super::*;
use crate::dreamer::ObservationKind;
use meganeura::graph::Op;
use std::collections::HashSet;

fn ancestors(g: &Graph, id: NodeId) -> HashSet<NodeId> {
    let mut result = HashSet::new();
    let mut pending = vec![id];
    while let Some(id) = pending.pop() {
        if result.insert(id) {
            pending.extend(g.node(id).inputs.iter().copied());
        }
    }
    result
}

#[test]
fn posterior_generated_images_use_current_posterior_and_only_earlier_generated_history() {
    let mut c = DreamerConfig::tiny(3);
    c.batch_size = 2;
    c.observation_kind = ObservationKind::Rgb32;
    c.rgb_spatial_decoder = true;
    c.rgb_spatial_fusion = true;
    c.rgb_spatial_action = true;
    let mut g = Graph::new();
    let model = WorldModel::with_behavior(&mut g, &c, false);
    let width = c.observation_dim();
    let b = c.batch_size;
    let states = (0..2)
        .map(|t| g.input(&format!("posterior{t}"), &[b, c.feature_dim()]))
        .collect::<Vec<_>>();
    let actions = (0..2)
        .map(|t| g.input(&format!("action{t}"), &[b, c.action_count]))
        .collect::<Vec<_>>();
    let images = (0..3)
        .map(|t| g.input(&format!("target{t}"), &[b, width]))
        .collect::<Vec<_>>();
    let keeps = (0..2)
        .map(|t| g.input(&format!("keep{t}"), &[b, 1]))
        .collect::<Vec<_>>();
    let history = g.input("seed_history", &[b, 2 * width]);
    let rollout = build(
        &mut g,
        &c,
        &model,
        Sequence {
            posterior: &states,
            actions: &actions,
            observations: &images,
            keeps: &keeps,
            history,
        },
    );
    assert!(rollout.events.is_none());
    assert_eq!(rollout.states, states);
    for (t, &prediction) in rollout.predictions.iter().enumerate() {
        let seen = ancestors(&g, prediction);
        assert!(seen.contains(&states[t]) && seen.contains(&actions[t]));
        assert!(images.iter().all(|v| !seen.contains(v)));
        if t == 0 {
            assert!(!seen.contains(&states[1]) && !seen.contains(&actions[1]));
        }
        if t > 0 {
            assert!(seen.contains(&rollout.predictions[t - 1]));
        }
        assert!(!seen.iter().any(|&id|matches!(&g.node(id).op,Op::Input{name} if name.starts_with("pixel_rollout_uniforms")||name.starts_with("pixel_rollout_weight"))));
    }
    let seen = ancestors(&g, rollout.loss);
    assert!(
        rollout.predictions[..rollout.predictions.len() - 1]
            .iter()
            .all(|id| seen.contains(id))
    );
    g.set_outputs(vec![rollout.loss]);
    let (plan, _) = meganeura::compile_training_graph(&g);
    assert!(
        plan.param_buffers
            .iter()
            .filter(|(name, _)| name.starts_with("world.decoder."))
            .all(|(_, p)| plan.param_grad_pairs.iter().any(|(q, _)| p == q))
    );
}
