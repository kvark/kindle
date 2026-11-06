use meganeura::{Graph, graph::NodeId, nn};

use crate::dreamer::{config::DreamerConfig, networks::MlpHead};

pub(crate) struct Actor {
    pub(super) base: MlpHead,
    residual: Option<nn::Linear>,
}

impl Actor {
    pub(crate) fn new(graph: &mut Graph, config: &DreamerConfig) -> Self {
        let base = MlpHead::new(
            graph,
            "behavior.actor",
            config.feature_dim(),
            config.actor_hidden_units(),
            3,
            config.action_count,
        );
        let residual = config.actor_residual.then(|| {
            nn::Linear::new(
                graph,
                "behavior.actor.residual",
                config.feature_dim(),
                config.action_count,
            )
        });
        Self { base, residual }
    }

    fn combine(&self, graph: &mut Graph, input: NodeId, base: NodeId) -> NodeId {
        match &self.residual {
            Some(residual) => {
                let correction = residual.forward(graph, input);
                graph.add(base, correction)
            }
            None => base,
        }
    }

    pub(crate) fn forward(&self, graph: &mut Graph, input: NodeId) -> NodeId {
        let base = self.base.forward(graph, input);
        self.combine(graph, input, base)
    }

    pub(super) fn forward_policy(&self, graph: &mut Graph, input: NodeId) -> NodeId {
        let mut base = self.base.forward(graph, input);
        if self.residual.is_some() {
            base = graph.stop_gradient(base);
        }
        self.combine(graph, input, base)
    }
}
