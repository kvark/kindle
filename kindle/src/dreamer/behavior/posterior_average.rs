use super::*;
use crate::dreamer::networks::{feature, gumbel_sample, repeat_rows, slice_columns};

/// Frozen conditional action probabilities; recurrent state remains sampled once.
pub(crate) fn build_actor_posterior_average_graph(
    config: &DreamerConfig,
    batch: usize,
    samples: usize,
) -> Graph {
    config.validate();
    assert!(batch > 0 && (2..=16).contains(&samples));
    let size = config.network();
    let rows = batch.checked_mul(samples).expect("actor batch overflow");
    let latent = size.stoch * size.classes;
    let mut graph = Graph::new();
    let actor = Actor::new(&mut graph, config);
    let state = graph.input("feature", &[batch, config.feature_dim()]);
    let posterior = graph.input("posterior_logits", &[batch, latent]);
    let uniforms = graph.input("posterior_uniforms", &[rows * size.stoch, size.classes]);
    let deter = slice_columns(
        &mut graph,
        state,
        batch,
        config.feature_dim(),
        0,
        size.deter,
    );
    let deter = repeat_rows(&mut graph, deter, samples);
    let posterior = repeat_rows(&mut graph, posterior, samples);
    let posterior = graph.reshape(posterior, &[rows * size.stoch, size.classes]);
    let sampled = gumbel_sample(
        &mut graph,
        posterior,
        uniforms,
        rows * size.stoch,
        size.classes,
        config.unimix,
    );
    let states = feature(&mut graph, deter, sampled, rows, config);
    let logits = actor.forward(&mut graph, states);
    let probabilities = graph.softmax(logits);
    let probabilities = graph.reshape(probabilities, &[batch, samples, config.action_count]);
    let probabilities = graph.transpose(probabilities);
    let probabilities = graph.reshape(probabilities, &[batch * config.action_count, samples]);
    let total = graph.sum_inner(probabilities);
    let mean = graph.scale(total, 1.0 / samples as f32);
    let mean = graph.reshape(mean, &[batch, config.action_count]);
    let bounded = graph.clamp(mean, 1e-30, 1.0);
    let logits = graph.log(bounded);
    graph.set_outputs(vec![logits]);
    graph
}
