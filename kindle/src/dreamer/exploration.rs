//! Action-conditioned latent disagreement, evaluated with current parameters.

use meganeura::{Graph, graph::NodeId};

use super::config::DreamerConfig;
use super::networks::{MlpHead, concat_columns, sum};

const MEMBERS: usize = 4;

#[cfg(test)]
mod tests;

pub(super) struct Disagreement {
    heads: Vec<MlpHead>,
    feature_dim: usize,
    actions: usize,
}

impl Disagreement {
    pub(super) fn new(graph: &mut Graph, config: &DreamerConfig) -> Self {
        let size = config.network();
        Self {
            heads: (0..MEMBERS)
                .map(|member| {
                    MlpHead::new(
                        graph,
                        &format!("world.exploration.head{member}"),
                        config.feature_dim() + config.action_count,
                        size.units,
                        2,
                        size.stoch * size.classes,
                    )
                })
                .collect(),
            feature_dim: config.feature_dim(),
            actions: config.action_count,
        }
    }

    fn predict(&self, graph: &mut Graph, state: NodeId, action: NodeId) -> Vec<NodeId> {
        let rows = graph.node(state).ty.shape[0];
        let state = graph.stop_gradient(state);
        let action = graph.stop_gradient(action);
        let input = concat_columns(graph, state, action, rows, self.feature_dim, self.actions);
        self.heads
            .iter()
            .map(|head| head.forward(graph, input))
            .collect()
    }

    pub(super) fn loss(
        &self,
        graph: &mut Graph,
        state: NodeId,
        action: NodeId,
        target: NodeId,
        weight: NodeId,
    ) -> NodeId {
        let predictions = self.predict(graph, state, action);
        masked_loss(graph, &predictions, target, weight)
    }

    pub(super) fn bonus(&self, graph: &mut Graph, state: NodeId, action: NodeId) -> NodeId {
        let predictions = self.predict(graph, state, action);
        disagreement(graph, &predictions)
    }
}

fn masked_loss(
    graph: &mut Graph,
    predictions: &[NodeId],
    target: NodeId,
    weight: NodeId,
) -> NodeId {
    let width = graph.node(target).ty.shape[1];
    let target = graph.stop_gradient(target);
    let negative = graph.neg(target);
    let losses = predictions
        .iter()
        .map(|&prediction| {
            let error = graph.add(prediction, negative);
            let square = graph.mul(error, error);
            let per_row = graph.sum_inner(square);
            let weighted = graph.mul(per_row, weight);
            graph.mean_all(weighted)
        })
        .collect::<Vec<_>>();
    let loss = sum(graph, &losses);
    graph.scale(loss, 1.0 / (predictions.len() * width) as f32)
}

fn disagreement(graph: &mut Graph, predictions: &[NodeId]) -> NodeId {
    assert!(predictions.len() > 1);
    let width = graph.node(predictions[0]).ty.shape[1];
    let total = sum(graph, predictions);
    let negative_mean = graph.scale(total, -1.0 / predictions.len() as f32);
    let squares = predictions
        .iter()
        .map(|&prediction| {
            let centered = graph.add(prediction, negative_mean);
            graph.mul(centered, centered)
        })
        .collect::<Vec<_>>();
    let total = sum(graph, &squares);
    let variance = graph.scale(total, 1.0 / predictions.len() as f32);
    // Smooth sqrt, subtracting its floor so identical heads receive no bonus.
    let epsilon = graph.constant(vec![1e-8; width], &[width]);
    let bounded = graph.bias_add(variance, epsilon);
    let log = graph.log(bounded);
    let half_log = graph.scale(log, 0.5);
    let root = graph.exp(half_log);
    let floor = graph.constant(vec![-1e-4; width], &[width]);
    let root = graph.bias_add(root, floor);
    let root = graph.relu(root);
    let total = graph.sum_inner(root);
    graph.scale(total, 1.0 / width as f32)
}
