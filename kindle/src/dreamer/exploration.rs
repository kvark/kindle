//! Disagreement about action effects on detached observation encodings.

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
                        config.encoded_observation_dim(),
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
        let rows = graph.node(state).ty.shape[0];
        let states = repeat_rows(graph, state, self.actions);
        let actions = graph.constant(
            (0..rows * self.actions * self.actions)
                .map(|i| f32::from(i / self.actions % self.actions == i % self.actions))
                .collect(),
            &[rows * self.actions, self.actions],
        );
        let predictions = self.predict(graph, states, actions);
        let bonus = action_effect_disagreement(graph, &predictions, self.actions);
        let bonus = graph.reshape(bonus, &[rows, self.actions]);
        let action = graph.stop_gradient(action);
        let selected = graph.mul(bonus, action);
        graph.sum_inner(selected)
    }
}

fn repeat_rows(graph: &mut Graph, input: NodeId, repeats: usize) -> NodeId {
    let shape = graph.node(input).ty.shape.clone();
    let input = graph.reshape(input, &[shape[0] * shape[1], 1]);
    let input = graph.broadcast_inner(input, repeats);
    let input = graph.reshape(input, &[shape[0], shape[1], repeats]);
    let input = graph.transpose(input);
    graph.reshape(input, &[shape[0] * repeats, shape[1]])
}

fn action_effect_disagreement(graph: &mut Graph, predictions: &[NodeId], actions: usize) -> NodeId {
    let shape = graph.node(predictions[0]).ty.shape.clone();
    assert_eq!(shape[0] % actions, 0);
    let (rows, width) = (shape[0] / actions, shape[1]);
    let effects = predictions
        .iter()
        .map(|&prediction| {
            let by_action = graph.reshape(prediction, &[rows, actions, width]);
            let by_coordinate = graph.transpose(by_action);
            let by_coordinate = graph.reshape(by_coordinate, &[rows * width, actions]);
            let sum = graph.sum_inner(by_coordinate);
            let negative_mean = graph.scale(sum, -1.0 / actions as f32);
            let negative_mean = graph.reshape(negative_mean, &[rows, width]);
            let negative_mean = repeat_rows(graph, negative_mean, actions);
            graph.add(prediction, negative_mean)
        })
        .collect::<Vec<_>>();
    disagreement(graph, &effects)
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
