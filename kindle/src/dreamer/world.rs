//! Categorical DreamerV3 world model graphs.

use meganeura::{Graph, graph::NodeId};

use super::behavior;
use super::config::VideoEncoder;
use super::config::{DreamerConfig, FeatureStandardization, ObservationKind};
use super::exploration::Disagreement;
use super::networks::{
    FeatureDecoder, MlpHead, ObservationDecoder, Prior, Representation, RssmCore, categorical_kl,
    feature, gumbel_sample, mixed_probabilities, scale, slice_columns, straight_through_sample,
    sum, weighted_cross_entropy,
};
use crate::vision::levjepa::joint;
#[cfg(test)]
use crate::vision::{OBSERVATION_CHANNELS, OBSERVATION_GRID};

pub const LOSS_TOTAL: usize = 0;
pub const LOSS_RECONSTRUCTION: usize = 1;
pub const LOSS_DYNAMICS: usize = 2;
pub const LOSS_REPRESENTATION: usize = 3;
pub const LOSS_REWARD: usize = 4;
pub const LOSS_CONTINUATION: usize = 5;
pub const LOSS_REPLAY_VALUE: usize = 6;
pub const RAW_KL: usize = 7;
pub const LOSS_FUTURE_PREDICTION: usize = 8;
pub const LOSS_ENCODER_REGULARIZATION: usize = 9;
pub const ENCODER_SPREAD: usize = 10;
pub const LOSS_INITIAL_POLICY: usize = 11;
pub const LOSS_INITIAL_VALUE: usize = 12;
pub const LOSS_EXPLORATION: usize = 13;
pub const FUTURE_HEAD_REVISION: &str = "spatial-deterministic-v1";

mod cdp;

pub(crate) fn future_head_revision(config: &DreamerConfig) -> Option<&'static str> {
    (config.loss_scales.future_prediction > 0.0).then_some(if config.is_cdp() {
        "continuous-cosine-v1"
    } else {
        FUTURE_HEAD_REVISION
    })
}

pub const IMAGINATION_FEATURE: usize = 0;
pub const IMAGINATION_ALL_FEATURES: usize = 1;
pub const IMAGINATION_ACTION: usize = 2;
pub const IMAGINATION_REWARD: usize = 3;
pub const IMAGINATION_CONTINUATION: usize = 4;
pub const IMAGINATION_VALUE: usize = 5;
pub const IMAGINATION_BONUS: usize = 6;
pub const POSTERIOR_BONUS: usize = 2;

fn standardized_target(
    graph: &mut Graph,
    target: NodeId,
    statistics: Option<&FeatureStandardization>,
) -> NodeId {
    let Some(stats) = statistics else {
        return target;
    };
    let width = stats.mean.len();
    let rows = graph.node(target).ty.shape.iter().product::<usize>() / width;
    let target = graph.reshape(target, &[rows, width]);
    let negative_mean = graph.constant(stats.mean.iter().map(|x| -x).collect(), &[width]);
    let inverse_scale = graph.constant(stats.scale.iter().map(|x| x.recip()).collect(), &[width]);
    let centered = graph.bias_add(target, negative_mean);
    graph.bias_mul(centered, inverse_scale)
}

fn raw_prediction(
    graph: &mut Graph,
    prediction: NodeId,
    statistics: Option<&FeatureStandardization>,
) -> NodeId {
    let Some(stats) = statistics else {
        return prediction;
    };
    let shape = graph.node(prediction).ty.shape.clone();
    let width = stats.mean.len();
    let rows = shape.iter().product::<usize>() / width;
    let prediction = graph.reshape(prediction, &[rows, width]);
    let scale = graph.constant(stats.scale.clone(), &[width]);
    let mean = graph.constant(stats.mean.clone(), &[width]);
    let scaled = graph.bias_mul(prediction, scale);
    let raw = graph.bias_add(scaled, mean);
    graph.reshape(raw, &shape)
}

fn replay_observations(graph: &mut Graph, config: &DreamerConfig, length: usize) -> Vec<NodeId> {
    let batch = config.batch_size;
    if let Some(mode) = config.video_encoder {
        let pixels = (0..length)
            .map(|time| graph.input(&format!("pixels_{time}"), &[batch, joint::PIXELS]))
            .collect::<Vec<_>>();
        let features = joint::encode(graph, &pixels, batch);
        if mode == VideoEncoder::Frozen {
            features
                .into_iter()
                .map(|x| graph.stop_gradient(x))
                .collect()
        } else {
            features
        }
    } else {
        (0..length)
            .map(|time| {
                graph.input(
                    &format!("observation_{time}"),
                    &config.observation_shape(batch),
                )
            })
            .collect()
    }
}

struct Dynamics {
    core: RssmCore,
    prior: Prior,
}

impl Dynamics {
    fn new(graph: &mut Graph, config: &DreamerConfig) -> Self {
        Self {
            core: RssmCore::new(graph, config),
            prior: Prior::new(graph, config),
        }
    }
}

struct WorldHeads {
    reward: MlpHead,
    continuation: MlpHead,
}

impl WorldHeads {
    fn new(graph: &mut Graph, config: &DreamerConfig) -> Self {
        let units = config.network().units;
        Self {
            reward: MlpHead::new(
                graph,
                "world.reward",
                config.feature_dim(),
                units,
                1,
                config.value_bins,
            ),
            continuation: MlpHead::new(
                graph,
                "world.continuation",
                config.feature_dim(),
                units,
                1,
                1,
            ),
        }
    }

    fn forward(&self, graph: &mut Graph, state: NodeId) -> (NodeId, NodeId) {
        let reward = self.reward.forward(graph, state);
        let continuation = self.continuation.forward(graph, state);
        let continuation = graph.sigmoid(continuation);
        (reward, continuation)
    }
}

struct WorldModel {
    dynamics: Dynamics,
    representation: Representation,
    decoder: Option<ObservationDecoder>,
    future_predictor: Option<FuturePredictor>,
    heads: WorldHeads,
    /// The behavior optimizer owns these parameters. They are frozen in this
    /// graph so replay-value gradients only shape the posterior/RSSM path.
    replay_value: MlpHead,
    actor: Option<MlpHead>,
    exploration: Option<Disagreement>,
}

impl WorldModel {
    fn new(graph: &mut Graph, config: &DreamerConfig) -> Self {
        let units = config.network().units;
        Self {
            dynamics: Dynamics::new(graph, config),
            representation: Representation::new(graph, config),
            decoder: (config.loss_scales.reconstruction > 0.0).then(|| {
                ObservationDecoder::new(graph, config, "world.decoder", config.feature_dim())
            }),
            future_predictor: (config.loss_scales.future_prediction > 0.0)
                .then(|| future_predictor(graph, config)),
            heads: WorldHeads::new(graph, config),
            exploration: config
                .uses_disagreement()
                .then(|| Disagreement::new(graph, config)),
            replay_value: MlpHead::new(
                graph,
                "behavior.value",
                config.feature_dim(),
                units,
                3,
                config.value_bins,
            ),
            actor: config.actor_critic_gradient.then(|| {
                MlpHead::new(
                    graph,
                    "behavior.actor",
                    config.feature_dim(),
                    units,
                    3,
                    config.action_count,
                )
            }),
        }
    }
}

enum FuturePredictor {
    Features(FeatureDecoder),
    Cdp(MlpHead),
}

impl FuturePredictor {
    fn forward(&self, graph: &mut Graph, deter: NodeId, batch: usize) -> NodeId {
        match self {
            Self::Features(head) => head.forward(graph, deter, batch),
            Self::Cdp(head) => head.forward(graph, deter),
        }
    }
}

fn future_predictor(graph: &mut Graph, config: &DreamerConfig) -> FuturePredictor {
    let name = "world.future_predictor";
    let input = config.network().deter;
    match config.observation_kind {
        ObservationKind::Features => {
            FuturePredictor::Features(FeatureDecoder::new(graph, config, name, input))
        }
        ObservationKind::Rgb64 => {
            let width = config.encoded_observation_dim();
            FuturePredictor::Cdp(MlpHead::new(graph, name, input, width, 1, width))
        }
    }
}

/// Full sequence loss with externally sampled hard posterior states.
///
/// The hard samples are produced immediately before this graph runs by the
/// observe graph. Inside this graph they are combined with posterior
/// probabilities using `hard + probs - stop_gradient(probs)`, exactly D3's
/// straight-through categorical estimator.
pub fn build_training_graph(config: &DreamerConfig, length: usize) -> Graph {
    build_training_graph_grouped(config, length, length)
}

// The RSSM and posterior stay sequential. Only row-independent operations are
// grouped across time; their RMS norms never mix rows. A one-step grouping is
// retained privately as the numerical reference for this execution change.
fn build_training_graph_grouped(
    config: &DreamerConfig,
    length: usize,
    time_batch_length: usize,
) -> Graph {
    config.validate();
    assert!(length > 0 && length <= config.batch_length);
    assert!(time_batch_length > 0 && length.is_multiple_of(time_batch_length));
    let batch = config.batch_size;
    let size = config.network();
    let mut graph = Graph::new();
    let model = WorldModel::new(&mut graph, config);
    let mut deter = graph.input("initial_deter", &[batch, size.deter]);
    let mut stoch = graph.input("initial_stoch", &[batch * size.stoch, size.classes]);

    let observations = replay_observations(&mut graph, config, length);
    let (encoder_regularization, encoder_spread) = if config.video_encoder.is_some() {
        joint::regularization(&mut graph, &observations, batch)
    } else {
        (graph.scalar(0.0), graph.scalar(0.0))
    };
    let mut encodings = Vec::with_capacity(length);
    for chunk in observations.chunks(time_batch_length) {
        let observation = stack_time(&mut graph, chunk, batch, config.observation_dim());
        let encoded =
            model
                .representation
                .encoder
                .forward(&mut graph, observation, batch * chunk.len());
        split_time(
            &mut graph,
            encoded,
            chunk.len(),
            batch,
            model.representation.encoder.output_dim(),
            &mut encodings,
        );
    }
    let mut head_inputs = Vec::with_capacity(length);
    let mut exploration_states = Vec::new();
    let mut exploration_actions = Vec::new();
    let mut exploration_targets = Vec::new();
    let mut exploration_weights = Vec::new();

    let mut reconstruction_losses = Vec::with_capacity(length);
    let mut future_prediction_losses = Vec::with_capacity(length);
    let mut dynamics_losses = Vec::with_capacity(length);
    let mut representation_losses = Vec::with_capacity(length);
    let mut raw_kl_metrics = Vec::with_capacity(length);
    let mut reward_losses = Vec::with_capacity(length);
    let mut continuation_losses = Vec::with_capacity(length);
    let mut replay_value_losses = Vec::with_capacity(length);
    let mut initial_policy_losses = Vec::new();
    let mut initial_value_losses = Vec::new();

    for time in 0..length {
        let action = graph.input(
            &format!("previous_action_{time}"),
            &[batch, config.action_count],
        );
        let keep_deter = graph.input(&format!("keep_deter_{time}"), &[batch, size.deter]);
        let keep_stoch = graph.input(
            &format!("keep_stoch_{time}"),
            &[batch * size.stoch, size.classes],
        );
        let keep_action = graph.input(
            &format!("keep_action_{time}"),
            &[batch, config.action_count],
        );
        let hard_sample = graph.input(
            &format!("posterior_sample_{time}"),
            &[batch * size.stoch, size.classes],
        );
        let masked_deter = graph.mul(deter, keep_deter);
        let masked_stoch = graph.mul(stoch, keep_stoch);
        let masked_action = graph.mul(action, keep_action);
        if model.exploration.is_some() {
            exploration_states.push(feature(
                &mut graph,
                masked_deter,
                masked_stoch,
                batch,
                config,
            ));
            exploration_actions.push(masked_action);
            let keep = graph.sum_inner(keep_action);
            exploration_weights.push(graph.scale(keep, 1.0 / config.action_count as f32));
        }
        deter = model.dynamics.core.forward(
            &mut graph,
            masked_deter,
            masked_stoch,
            masked_action,
            batch,
        );
        let prior_logits = model.dynamics.prior.forward(&mut graph, deter, batch);
        let posterior_logits =
            model
                .representation
                .posterior
                .forward(&mut graph, deter, encodings[time], batch);
        let probabilities = mixed_probabilities(
            &mut graph,
            posterior_logits,
            batch * size.stoch,
            size.classes,
            config.unimix,
        );
        stoch = straight_through_sample(&mut graph, hard_sample, probabilities);
        if model.exploration.is_some() {
            // The conditional mean preserves the ensemble's expected MSE
            // gradient without making it fit fresh categorical sampling noise.
            exploration_targets
                .push(graph.reshape(probabilities, &[batch, size.stoch * size.classes]));
        }

        let state = feature(&mut graph, deter, stoch, batch, config);
        head_inputs.push(HeadInputs {
            observation: if config.is_cdp() {
                encodings[time]
            } else {
                observations[time]
            },
            deter,
            state,
            keep_action,
        });

        let dynamics_kl = categorical_kl(
            &mut graph,
            posterior_logits,
            prior_logits,
            batch,
            size.stoch,
            size.classes,
            config.unimix,
            true,
            false,
            config.dynamics_free_nats(),
        );
        dynamics_losses.push(dynamics_kl.loss);
        raw_kl_metrics.push(dynamics_kl.raw);
        let representation_kl = categorical_kl(
            &mut graph,
            posterior_logits,
            prior_logits,
            batch,
            size.stoch,
            size.classes,
            config.unimix,
            false,
            true,
            config.free_nats,
        );
        representation_losses.push(representation_kl.loss);
    }

    let exploration_loss = if let Some(ensemble) = &model.exploration {
        let state = stack_time(&mut graph, &exploration_states, batch, config.feature_dim());
        let action = stack_time(&mut graph, &exploration_actions, batch, config.action_count);
        let target = stack_time(
            &mut graph,
            &exploration_targets,
            batch,
            size.stoch * size.classes,
        );
        let weight = stack_time(&mut graph, &exploration_weights, batch, 1);
        ensemble.loss(&mut graph, state, action, target, weight)
    } else {
        graph.scalar(0.0)
    };

    for (group, chunk) in head_inputs.chunks(time_batch_length).enumerate() {
        let rows = batch * chunk.len();
        let HeadInputs {
            observation,
            deter,
            state,
            keep_action,
        } = HeadInputs::stack(&mut graph, chunk, config);
        let mut target = |name, width| {
            let inputs = (group * time_batch_length..(group + 1) * time_batch_length)
                .map(|time| graph.input(&format!("{name}_{time}"), &[batch, width]))
                .collect::<Vec<_>>();
            stack_time(&mut graph, &inputs, batch, width)
        };
        let reward_target = target("reward_target", config.value_bins);
        let continuation_target = target("continuation_target", 1);
        let replay_value_target = target("replay_value_target", config.value_bins);
        let replay_slow_target = target("replay_slow_target", config.value_bins);
        let replay_value_weight = target("replay_value_weight", 1);
        let initial_targets = config.actor_critic_gradient.then(|| {
            (
                target("initial_action_target", config.action_count),
                target("initial_weight", 1),
                target("initial_value_target", config.value_bins),
                target("initial_slow_target", config.value_bins),
            )
        });
        let reward_weight = graph.constant(vec![1.0; rows], &[rows, 1]);

        if let Some(predictor) = &model.future_predictor {
            // Each row uses deter_t, before posterior_t consumes observation_t.
            let prediction = predictor.forward(&mut graph, deter, rows);
            let prediction = graph.reshape(prediction, &[rows, config.prediction_dim()]);
            let target = graph.stop_gradient(observation);
            let per_row = if config.is_cdp() {
                // Upstream CDP includes episode starts, predicting their
                // embeddings from the reset recurrent state (without pixels).
                cdp::cosine_distance(&mut graph, prediction, target)
            } else {
                let target = standardized_target(
                    &mut graph,
                    target,
                    config.future_target_standardization.as_ref(),
                );
                let negative_target = graph.neg(target);
                let residual = graph.add(prediction, negative_target);
                let squared = graph.mul(residual, residual);
                let per_row = graph.sum_inner(squared);
                // Reset observations have no preceding action-conditioned state.
                let keep = graph.sum_inner(keep_action);
                let normalization =
                    graph.constant(vec![1.0 / config.action_count as f32; rows], &[rows, 1]);
                let keep = graph.mul(keep, normalization);
                graph.mul(per_row, keep)
            };
            future_prediction_losses.push(graph.mean_all(per_row));
        }
        if let Some(decoder) = &model.decoder {
            let reconstruction = decoder.forward(&mut graph, state, rows);
            let reconstruction = graph.reshape(reconstruction, &[rows, config.observation_dim()]);
            let target = graph.stop_gradient(observation);
            let target = standardized_target(
                &mut graph,
                target,
                config.reconstruction_target_standardization.as_ref(),
            );
            let negative_target = graph.neg(target);
            let residual = graph.add(reconstruction, negative_target);
            let squared = graph.mul(residual, residual);
            let reconstruction_loss = graph.sum_all(squared);
            // Sum event dimensions and average B*T, as in the D3 control.
            reconstruction_losses.push(scale(&mut graph, reconstruction_loss, 1.0 / rows as f32));
        }

        let (reward, continuation) = model.heads.forward(&mut graph, state);
        // Keep the reduction explicit because this value is composed into the
        // joint loss and exported as a metric. The fused backend loss stores
        // per-row partials and is only scalar when used as a terminal output.
        reward_losses.push(weighted_cross_entropy(
            &mut graph,
            reward,
            reward_target,
            reward_weight,
        ));
        continuation_losses.push(mean_binary_cross_entropy(
            &mut graph,
            continuation,
            continuation_target,
            rows,
        ));
        let replay_value_state = if config.replay_value_gradient {
            state
        } else {
            graph.stop_gradient(state)
        };
        let value = model
            .replay_value
            .forward_frozen(&mut graph, replay_value_state);
        let value_target =
            weighted_cross_entropy(&mut graph, value, replay_value_target, replay_value_weight);
        let slow_target =
            weighted_cross_entropy(&mut graph, value, replay_slow_target, replay_value_weight);
        replay_value_losses.push(graph.add(value_target, slow_target));
        if let (Some(actor), Some((action_target, weight, value_target, slow_target))) =
            (&model.actor, initial_targets)
        {
            // Same stopped targets and frozen heads as behavior training. Only
            // imagination t=0 is a posterior state; later states are detached.
            let logits = actor.forward_frozen(&mut graph, state);
            let (policy, _, _) =
                behavior::policy_loss(&mut graph, config, logits, action_target, weight, rows);
            let value = model.replay_value.forward_frozen(&mut graph, state);
            initial_policy_losses.push(policy);
            initial_value_losses.push(behavior::value_loss(
                &mut graph,
                value,
                value_target,
                slow_target,
                weight,
            ));
        }
    }

    let average = 1.0 / length as f32;
    let head_average = time_batch_length as f32 * average;
    let reconstruction = sum_or_zero(&mut graph, &reconstruction_losses);
    let reconstruction = scale(&mut graph, reconstruction, head_average);
    let future_prediction = sum_or_zero(&mut graph, &future_prediction_losses);
    let future_prediction = scale(&mut graph, future_prediction, head_average);
    let dynamics = sum(&mut graph, &dynamics_losses);
    let dynamics = scale(&mut graph, dynamics, average);
    let representation = sum(&mut graph, &representation_losses);
    let representation = scale(&mut graph, representation, average);
    let raw_kl = sum(&mut graph, &raw_kl_metrics);
    let raw_kl = scale(&mut graph, raw_kl, average);
    let reward = sum(&mut graph, &reward_losses);
    let reward = scale(&mut graph, reward, head_average);
    let continuation = sum(&mut graph, &continuation_losses);
    let continuation = scale(&mut graph, continuation, head_average);
    let replay_value = sum(&mut graph, &replay_value_losses);
    let replay_value = scale(&mut graph, replay_value, head_average);

    let scales = config.loss_scales;
    let weighted = [
        scale(&mut graph, reconstruction, scales.reconstruction),
        scale(&mut graph, future_prediction, scales.future_prediction),
        scale(&mut graph, dynamics, scales.dynamics),
        scale(&mut graph, representation, scales.representation),
        scale(&mut graph, reward, scales.reward),
        scale(&mut graph, continuation, scales.continuation),
        scale(&mut graph, replay_value, scales.replay_value),
    ];
    let total = sum(&mut graph, &weighted);
    let total = if model.exploration.is_some() {
        graph.add(total, exploration_loss)
    } else {
        total
    };
    let mut initial_losses = None;
    let total = if config.actor_critic_gradient {
        // Upstream averages over H imagined losses. The other H-1 states
        // contribute behavior gradients but no world/encoder gradients.
        let average = head_average / config.imagination_length as f32;
        let policy = sum(&mut graph, &initial_policy_losses);
        let policy = scale(&mut graph, policy, average);
        let value = sum(&mut graph, &initial_value_losses);
        let value = scale(&mut graph, value, average);
        initial_losses = Some((policy, value));
        let policy = scale(&mut graph, policy, scales.policy);
        let actor_scale = graph.input("actor_update_scale", &[1]);
        let policy = graph.mul(policy, actor_scale);
        let value = scale(&mut graph, value, scales.value);
        sum(&mut graph, &[total, policy, value])
    } else {
        total
    };
    let total = if config.video_encoder.is_some() {
        let regularizer = scale(
            &mut graph,
            encoder_regularization,
            joint::REGULARIZATION_WEIGHT,
        );
        graph.add(total, regularizer)
    } else {
        total
    };
    let mut outputs = vec![
        total,
        reconstruction,
        dynamics,
        representation,
        reward,
        continuation,
        replay_value,
        raw_kl,
        future_prediction,
    ];
    if config.video_encoder.is_some() || config.actor_critic_gradient || config.uses_disagreement()
    {
        outputs.extend([encoder_regularization, encoder_spread]);
    }
    if let Some((policy, value)) = initial_losses {
        outputs.extend([policy, value]);
    } else if config.uses_disagreement() {
        outputs.extend([graph.scalar(0.0), graph.scalar(0.0)]);
    }
    if config.uses_disagreement() {
        outputs.push(exploration_loss);
    }
    graph.set_outputs(outputs);
    graph
}

struct HeadInputs {
    observation: NodeId,
    deter: NodeId,
    state: NodeId,
    keep_action: NodeId,
}

fn mean_binary_cross_entropy(
    graph: &mut Graph,
    prediction: NodeId,
    target: NodeId,
    rows: usize,
) -> NodeId {
    // The backend BCE kernel emits one partial per 256 rows, not a final
    // scalar reduction. Compose only scalar-sized pieces into the joint loss.
    let mut losses = Vec::with_capacity(rows.div_ceil(256));
    for start in (0..rows).step_by(256) {
        let count = (rows - start).min(256);
        let prediction = slice_columns(graph, prediction, 1, rows, start, count);
        let target = slice_columns(graph, target, 1, rows, start, count);
        let loss = graph.bce_loss(prediction, target);
        losses.push(scale(graph, loss, count as f32 / rows as f32));
    }
    sum(graph, &losses)
}

impl HeadInputs {
    fn stack(graph: &mut Graph, steps: &[Self], config: &DreamerConfig) -> Self {
        let mut stack = |field: fn(&Self) -> NodeId, width| {
            let values = steps.iter().map(field).collect::<Vec<_>>();
            stack_time(graph, &values, config.batch_size, width)
        };
        Self {
            observation: stack(|step| step.observation, config.prediction_dim()),
            deter: stack(|step| step.deter, config.network().deter),
            state: stack(|step| step.state, config.feature_dim()),
            keep_action: stack(|step| step.keep_action, config.action_count),
        }
    }
}

// Balanced trees avoid quadratic copying/padding in forward and backward.
// Row order is time-major throughout: all B rows at t, then all B rows at t+1.
fn stack_time(graph: &mut Graph, steps: &[NodeId], batch: usize, width: usize) -> NodeId {
    assert!(!steps.is_empty());
    if steps.len() == 1 {
        return graph.reshape(steps[0], &[batch, width]);
    }
    let middle = steps.len() / 2;
    let left = stack_time(graph, &steps[..middle], batch, width);
    let right = stack_time(graph, &steps[middle..], batch, width);
    let left_rows = middle * batch;
    let right_rows = (steps.len() - middle) * batch;
    let left = graph.reshape(left, &[left_rows * width]);
    let right = graph.reshape(right, &[right_rows * width]);
    let joined = graph.concat(
        left,
        right,
        1,
        left_rows as u32,
        right_rows as u32,
        width as u32,
    );
    graph.reshape(joined, &[steps.len() * batch, width])
}

fn split_time(
    graph: &mut Graph,
    input: NodeId,
    length: usize,
    batch: usize,
    width: usize,
    output: &mut Vec<NodeId>,
) {
    assert!(length > 0);
    if length == 1 {
        output.push(graph.reshape(input, &[batch, width]));
        return;
    }
    let middle = length / 2;
    let left_rows = middle * batch;
    let right_rows = (length - middle) * batch;
    let input = graph.reshape(input, &[length * batch * width]);
    let left = graph.split_a(input, 1, left_rows as u32, right_rows as u32, width as u32);
    let right = graph.split_b(input, 1, left_rows as u32, right_rows as u32, width as u32);
    split_time(graph, left, middle, batch, width, output);
    split_time(graph, right, length - middle, batch, width, output);
}

fn sum_or_zero(graph: &mut Graph, values: &[NodeId]) -> NodeId {
    if values.is_empty() {
        graph.constant(vec![0.0], &[1])
    } else {
        sum(graph, values)
    }
}

/// Unrolled posterior sampling. CPU draws and sequence inputs are installed
/// before submission; recurrence and categorical sampling never visit the host.
pub fn build_posterior_graph(config: &DreamerConfig) -> Graph {
    config.validate();
    let batch = config.batch_size;
    let length = config.batch_length;
    let size = config.network();
    let mut graph = Graph::new();
    let core = RssmCore::new(&mut graph, config);
    let representation = Representation::new(&mut graph, config);
    let exploration = config
        .uses_disagreement()
        .then(|| Disagreement::new(&mut graph, config));
    let mut deter = graph.input("initial_deter", &[batch, size.deter]);
    let mut stoch = graph.input("initial_stoch", &[batch * size.stoch, size.classes]);
    let observations = replay_observations(&mut graph, config, length);
    let observations = stack_time(&mut graph, &observations, batch, config.observation_dim());
    let encoded = representation
        .encoder
        .forward(&mut graph, observations, batch * length);
    let mut encodings = Vec::with_capacity(length);
    split_time(
        &mut graph,
        encoded,
        length,
        batch,
        representation.encoder.output_dim(),
        &mut encodings,
    );
    let mut deters = Vec::with_capacity(length);
    let mut stochs = Vec::with_capacity(length);
    let mut exploration_states = Vec::new();
    let mut exploration_actions = Vec::new();
    let mut exploration_weights = Vec::new();
    for (time, encoded) in encodings.into_iter().enumerate() {
        let action = graph.input(
            &format!("previous_action_{time}"),
            &[batch, config.action_count],
        );
        let keep_deter = graph.input(&format!("keep_deter_{time}"), &[batch, size.deter]);
        let keep_stoch = graph.input(
            &format!("keep_stoch_{time}"),
            &[batch * size.stoch, size.classes],
        );
        let keep_action = graph.input(
            &format!("keep_action_{time}"),
            &[batch, config.action_count],
        );
        let uniforms = graph.input(
            &format!("uniforms_{time}"),
            &[batch * size.stoch, size.classes],
        );
        let previous_deter = graph.mul(deter, keep_deter);
        let previous_stoch = graph.mul(stoch, keep_stoch);
        let action = graph.mul(action, keep_action);
        if exploration.is_some() {
            exploration_states.push(feature(
                &mut graph,
                previous_deter,
                previous_stoch,
                batch,
                config,
            ));
            exploration_actions.push(action);
            let keep = graph.sum_inner(keep_action);
            exploration_weights.push(graph.scale(keep, 1.0 / config.action_count as f32));
        }
        deter = core.forward(&mut graph, previous_deter, previous_stoch, action, batch);
        let logits = representation
            .posterior
            .forward(&mut graph, deter, encoded, batch);
        stoch = gumbel_sample(
            &mut graph,
            logits,
            uniforms,
            batch * size.stoch,
            size.classes,
            config.unimix,
        );
        deters.push(deter);
        stochs.push(stoch);
    }
    let deter = stack_time(&mut graph, &deters, batch, size.deter);
    let stoch = stack_time(&mut graph, &stochs, batch, size.stoch * size.classes);
    let mut outputs = vec![deter, stoch];
    if let Some(ensemble) = exploration {
        let state = stack_time(&mut graph, &exploration_states, batch, config.feature_dim());
        let action = stack_time(&mut graph, &exploration_actions, batch, config.action_count);
        let weight = stack_time(&mut graph, &exploration_weights, batch, 1);
        let bonus = ensemble.bonus(&mut graph, state, action);
        outputs.push(graph.mul(bonus, weight));
    }
    graph.set_outputs(outputs);
    graph
}

/// Unrolled prior/actor recurrence, followed by time-batched reward/value heads.
/// The independent slow critic consumes ALL_FEATURES in a separate GPU session.
pub fn build_imagination_graph(config: &DreamerConfig) -> Graph {
    config.validate();
    let rows = config.batch_size * config.batch_length;
    let horizon = config.imagination_length;
    let size = config.network();
    let mut graph = Graph::new();
    let dynamics = Dynamics::new(&mut graph, config);
    let heads = WorldHeads::new(&mut graph, config);
    let actor = MlpHead::new(
        &mut graph,
        "behavior.actor",
        config.feature_dim(),
        size.units,
        3,
        config.action_count,
    );
    let value = MlpHead::new(
        &mut graph,
        "behavior.value",
        config.feature_dim(),
        size.units,
        3,
        config.value_bins,
    );
    let mut deter = graph.input("deter", &[rows, size.deter]);
    let mut stoch = graph.input("stoch", &[rows * size.stoch, size.classes]);
    let mut features = Vec::with_capacity(horizon + 1);
    let mut actions = Vec::with_capacity(horizon);
    for time in 0..horizon {
        let state = feature(&mut graph, deter, stoch, rows, config);
        features.push(state);
        let logits = actor.forward(&mut graph, state);
        let uniforms = graph.input(
            &format!("action_uniforms_{time}"),
            &[rows, config.action_count],
        );
        let action = gumbel_sample(
            &mut graph,
            logits,
            uniforms,
            rows,
            config.action_count,
            config.actor_unimix,
        );
        actions.push(action);
        deter = dynamics
            .core
            .forward(&mut graph, deter, stoch, action, rows);
        let logits = dynamics.prior.forward(&mut graph, deter, rows);
        let uniforms = graph.input(
            &format!("latent_uniforms_{time}"),
            &[rows * size.stoch, size.classes],
        );
        stoch = gumbel_sample(
            &mut graph,
            logits,
            uniforms,
            rows * size.stoch,
            size.classes,
            config.unimix,
        );
    }
    features.push(feature(&mut graph, deter, stoch, rows, config));
    let imagined = stack_time(&mut graph, &features[..horizon], rows, config.feature_dim());
    let all = stack_time(&mut graph, &features, rows, config.feature_dim());
    let actions = stack_time(&mut graph, &actions, rows, config.action_count);
    let (reward, continuation) = heads.forward(&mut graph, all);
    let values = value.forward(&mut graph, all);
    let mut outputs = vec![imagined, all, actions, reward, continuation, values];
    if config.uses_disagreement() {
        let ensemble = Disagreement::new(&mut graph, config);
        outputs.push(ensemble.bonus(&mut graph, imagined, actions));
    }
    graph.set_outputs(outputs);
    graph
}

/// One posterior update used by the live actor and the sequential test reference.
///
/// Outputs: next deterministic state, posterior logits, prior logits, and the
/// trainable observation-encoder output used by the posterior.
pub fn build_observe_graph(config: &DreamerConfig, batch: usize) -> Graph {
    config.validate();
    assert!(batch > 0);
    let size = config.network();
    let mut graph = Graph::new();
    let dynamics = Dynamics::new(&mut graph, config);
    let representation = Representation::new(&mut graph, config);
    let previous_deter = graph.input("previous_deter", &[batch, size.deter]);
    let previous_stoch = graph.input("previous_stoch", &[batch * size.stoch, size.classes]);
    let previous_action = graph.input("previous_action", &[batch, config.action_count]);
    let observation = graph.input("observation", &config.observation_shape(batch));
    let keep_deter = graph.input("keep_deter", &[batch, size.deter]);
    let keep_stoch = graph.input("keep_stoch", &[batch * size.stoch, size.classes]);
    let keep_action = graph.input("keep_action", &[batch, config.action_count]);
    let previous_deter = graph.mul(previous_deter, keep_deter);
    let previous_stoch = graph.mul(previous_stoch, keep_stoch);
    let previous_action = graph.mul(previous_action, keep_action);
    let deter = dynamics.core.forward(
        &mut graph,
        previous_deter,
        previous_stoch,
        previous_action,
        batch,
    );
    let prior = dynamics.prior.forward(&mut graph, deter, batch);
    let encoded = representation
        .encoder
        .forward(&mut graph, observation, batch);
    let posterior = representation
        .posterior
        .forward(&mut graph, deter, encoded, batch);
    graph.set_outputs(vec![deter, posterior, prior, encoded]);
    graph
}

/// One prior transition for imagination.
///
/// Outputs: next deterministic state and prior logits. Categorical sampling
/// happens on the CPU between this graph and the next imagined step.
pub fn build_transition_graph(config: &DreamerConfig, batch: usize) -> Graph {
    config.validate();
    assert!(batch > 0);
    let size = config.network();
    let mut graph = Graph::new();
    let dynamics = Dynamics::new(&mut graph, config);
    let deter = graph.input("deter", &[batch, size.deter]);
    let stoch = graph.input("stoch", &[batch * size.stoch, size.classes]);
    let action = graph.input("action", &[batch, config.action_count]);
    let next_deter = dynamics
        .core
        .forward(&mut graph, deter, stoch, action, batch);
    let prior = dynamics.prior.forward(&mut graph, next_deter, batch);
    graph.set_outputs(vec![next_deter, prior]);
    graph
}

/// Reward and continuation predictions for posterior or imagined states.
pub fn build_head_graph(config: &DreamerConfig, batch: usize) -> Graph {
    head_graph(config, batch, false)
}

/// Also expose the pinned state feature for device-side imagination handoffs.
#[cfg(test)]
pub fn build_imagination_head_graph(config: &DreamerConfig, batch: usize) -> Graph {
    head_graph(config, batch, true)
}

fn head_graph(config: &DreamerConfig, batch: usize, expose_state: bool) -> Graph {
    config.validate();
    assert!(batch > 0);
    let size = config.network();
    let mut graph = Graph::new();
    let heads = WorldHeads::new(&mut graph, config);
    let deter = graph.input("deter", &[batch, size.deter]);
    let stoch = graph.input("stoch", &[batch * size.stoch, size.classes]);
    let state = feature(&mut graph, deter, stoch, batch, config);
    let (reward, continuation) = heads.forward(&mut graph, state);
    let mut outputs = vec![reward, continuation];
    if expose_state {
        outputs.push(state);
    }
    graph.set_outputs(outputs);
    graph
}

/// Evaluate the future head when enabled, otherwise posterior reconstruction.
/// Forecasts read only the deterministic state, including in auxiliary runs.
/// This graph is constructed lazily by diagnostics.
pub fn build_observation_prediction_graph(config: &DreamerConfig, batch: usize) -> Graph {
    config.validate();
    assert!(batch > 0);
    let size = config.network();
    let mut graph = Graph::new();
    if config.loss_scales.future_prediction > 0.0 {
        let predictor = future_predictor(&mut graph, config);
        let deter = graph.input("deter", &[batch, size.deter]);
        graph.input("stoch", &[batch * size.stoch, size.classes]);
        let observation = predictor.forward(&mut graph, deter, batch);
        let observation = raw_prediction(
            &mut graph,
            observation,
            config.future_target_standardization.as_ref(),
        );
        graph.set_outputs(vec![observation]);
        return graph;
    }
    assert!(
        config.loss_scales.reconstruction > 0.0,
        "no observation prediction head"
    );
    let decoder =
        ObservationDecoder::new(&mut graph, config, "world.decoder", config.feature_dim());
    let deter = graph.input("deter", &[batch, size.deter]);
    let stoch = graph.input("stoch", &[batch * size.stoch, size.classes]);
    let state = feature(&mut graph, deter, stoch, batch, config);
    let observation = decoder.forward(&mut graph, state, batch);
    let observation = raw_prediction(
        &mut graph,
        observation,
        config.reconstruction_target_standardization.as_ref(),
    );
    graph.set_outputs(vec![observation]);
    graph
}

#[cfg(test)]
mod rollout_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_standardization_preserves_graph_nodes() {
        let config = DreamerConfig::tiny(3);
        let mut graph = Graph::new();
        let x = graph.input("x", &[2, config.observation_dim()]);
        assert_eq!(standardized_target(&mut graph, x, None), x);
        assert_eq!(raw_prediction(&mut graph, x, None), x);
    }

    #[test]
    #[ignore = "requires separately guarded GPU; independent full-width target standardization"]
    fn standardized_targets_match_f64_values_and_gradients() {
        use super::super::{config::FeatureStandardization, runtime::build_session};
        let mut config = DreamerConfig::tiny(3);
        config.loss_scales.future_prediction = 0.25;
        let width = config.observation_dim();
        let stats = FeatureStandardization {
            mean: (0..width).map(|i| 1.0 + (i % 17) as f32 * 0.1).collect(),
            scale: (0..width).map(|i| 0.03 + (i % 11) as f32 * 0.01).collect(),
        };
        config.future_target_standardization = Some(stats.clone());
        let mut graph = Graph::new();
        let p = graph.parameter("prediction", &[2, width]);
        let target = graph.input("target", &[2, width]);
        let target = graph.stop_gradient(target);
        let normalized = standardized_target(
            &mut graph,
            target,
            config.future_target_standardization.as_ref(),
        );
        let neg = graph.neg(normalized);
        let delta = graph.add(p, neg);
        let square = graph.mul(delta, delta);
        let loss = graph.sum_all(square);
        let loss = graph.scale(loss, 0.5);
        let raw = raw_prediction(&mut graph, p, config.future_target_standardization.as_ref());
        graph.set_outputs(vec![loss, normalized, raw]);
        let gpu = std::sync::Arc::new(crate::init_gpu_context().unwrap());
        assert_eq!(
            gpu.device_information().device_name,
            std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
        );
        let mut session = build_session(&graph, &gpu, meganeura::Mode::Training, false);
        let memory = session.device_memory_stats().unwrap();
        assert!(memory.budget_bytes.saturating_sub(memory.usage_bytes) >= 2 << 30);
        let predictions = (0..2 * width)
            .map(|i| (i % 29) as f32 * 0.02 - 0.3)
            .collect::<Vec<_>>();
        let targets = (0..2 * width)
            .map(|i| 1.9 + (i % 37) as f32 / 23.0)
            .collect::<Vec<_>>();
        session.set_parameter("prediction", &predictions);
        session.set_input("target", &targets);
        session.step();
        session.wait();
        let mut normalized = vec![0.0; 2 * width];
        let mut raw = normalized.clone();
        let mut gradients = normalized.clone();
        session.read_output_by_index(1, &mut normalized);
        session.read_output_by_index(2, &mut raw);
        session.read_param_grad("prediction", &mut gradients);
        let mut loss = 0.0_f64;
        let mut gradient_error = 0.0_f64;
        let mut gradient_norm = 0.0_f64;
        for i in 0..2 * width {
            let mean = f64::from(stats.mean[i % width]);
            let scale = f64::from(stats.scale[i % width]);
            let target = (f64::from(targets[i]) - mean) / scale;
            let p = f64::from(predictions[i]);
            assert!((f64::from(normalized[i]) - target).abs() < 2e-5);
            assert!((f64::from(raw[i]) - (p * scale + mean)).abs() < 1e-6);
            loss += 0.5 * (p - target).powi(2);
            gradient_error += (f64::from(gradients[i]) - (p - target)).powi(2);
            gradient_norm += (p - target).powi(2);
        }
        assert!((f64::from(session.read_loss()) / loss - 1.0).abs() < 1e-5);
        assert!((gradient_error / gradient_norm).sqrt() < 3e-6);
        assert_eq!(session.read_params(&["prediction"])[0], predictions);
        let memory = session.device_memory_stats().unwrap();
        assert!(memory.budget_bytes - memory.usage_bytes >= 2 << 30);
        eprintln!(
            "standardized full-width F64 values/gradients pass; relative-gradient-L2={}; memory={memory:?}",
            (gradient_error / gradient_norm).sqrt()
        );
    }

    #[test]
    fn task_losses_reach_every_joint_tiny_parameter_but_not_frozen_tiny() {
        for mode in [VideoEncoder::Frozen, VideoEncoder::Joint] {
            let mut config = DreamerConfig::tiny(3);
            config.batch_size = 1;
            config.batch_length = 16;
            config.world_backprop_length = 16;
            config.video_encoder = Some(mode);
            config.actor_critic_gradient = true;
            config.loss_scales.reconstruction = 0.0;
            config.loss_scales.future_prediction = 0.25;
            let mut graph = build_training_graph(&config, 16);
            let losses = graph.outputs().to_vec();
            // Test task supervision alone, not a regularizer disguising a
            // detached world/task path. This checks graph connectivity only.
            for loss in [
                LOSS_REWARD,
                LOSS_CONTINUATION,
                LOSS_REPLAY_VALUE,
                LOSS_FUTURE_PREDICTION,
                LOSS_ENCODER_REGULARIZATION,
                LOSS_INITIAL_POLICY,
                LOSS_INITIAL_VALUE,
            ] {
                graph.set_outputs(vec![losses[loss]]);
                let backward = meganeura::autodiff::differentiate(&graph);
                let mut checked = 0;
                for (parameter, &gradient) in graph
                    .nodes()
                    .iter()
                    .filter(|n| matches!(n.op, meganeura::graph::Op::Parameter { .. }))
                    .zip(&backward.outputs()[1..])
                {
                    if let meganeura::graph::Op::Parameter { ref name } = parameter.op {
                        let connected = backward.node(gradient).ty == parameter.ty;
                        if name.starts_with("encoder.") {
                            assert_eq!(
                                connected,
                                mode == VideoEncoder::Joint,
                                "{mode:?} loss{loss} {name}"
                            );
                            checked += 1;
                        } else if name.starts_with("behavior.") {
                            assert!(!connected, "world loss trained frozen behavior head {name}");
                        }
                    }
                }
                assert_eq!(checked, 148);
            }
        }
    }

    #[test]
    #[ignore = "requires separately guarded GPU and KINDLE_JOINT_TINY_REFERENCE"]
    fn isolated_policy_loss_updates_encoder_without_training_behavior_heads() {
        use super::super::runtime::{build_session, configure_d3_optimizer, initialize_d3};
        use meganeura::{Mode, data::safetensors::SafeTensorsModel, graph::Op};
        use std::{path::PathBuf, sync::Arc};

        let root = PathBuf::from(std::env::var_os("KINDLE_JOINT_TINY_REFERENCE").unwrap());
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
        let checkpoint = PathBuf::from(manifest["checkpoint"].as_str().unwrap());
        assert_eq!(
            crate::vision::checkpoint_sha256(&checkpoint).unwrap(),
            manifest["checkpoint_sha256"]
        );
        let weights = SafeTensorsModel::load(checkpoint).unwrap();
        let reference = SafeTensorsModel::load(root.join("reference.safetensors")).unwrap();
        let mut config = DreamerConfig::tiny(3);
        config.batch_size = 1;
        config.batch_length = 16;
        config.world_backprop_length = 16;
        config.video_encoder = Some(VideoEncoder::Joint);
        config.actor_critic_gradient = true;
        config.learning_rate_warmup = 0;
        config.loss_scales.reconstruction = 0.0;
        config.loss_scales.future_prediction = 0.25;
        let mut graph = build_training_graph(&config, 16);
        // No reward, value, KL, latent prediction or variance objective can
        // supply an encoder gradient in this test.
        graph.set_outputs(vec![graph.outputs()[LOSS_INITIAL_POLICY]]);
        let gpu = Arc::new(crate::init_gpu_context().unwrap());
        let check_device = || {
            let info = gpu.device_information();
            assert_eq!(
                info.device_name,
                std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
            );
            assert!(!info.is_software_emulated);
            let memory = gpu.memory_stats();
            assert!(memory.budget.saturating_sub(memory.usage) >= 2 << 30);
        };
        check_device();
        let mut session = build_session(&graph, &gpu, Mode::Training, false);
        initialize_d3(&mut session, &graph, 103);
        crate::vision::levjepa::load_weights(
            &mut session,
            &weights,
            0,
            crate::vision::levjepa::Architecture::Tiny,
        )
        .unwrap();
        for node in graph.nodes() {
            let Op::Input { name } = &node.op else {
                continue;
            };
            if session.input_buffer(name).is_none() {
                continue;
            }
            let count = node.ty.num_elements();
            let time = name
                .rsplit('_')
                .next()
                .unwrap()
                .parse::<usize>()
                .unwrap_or(0);
            let values = if name.starts_with("pixels_") {
                reference
                    .tensor_f32_auto(&format!("pixels_{}", time % 2))
                    .unwrap()[..count]
                    .to_vec()
            } else if name.starts_with("keep_")
                || name.starts_with("initial_weight_")
                || name == "actor_update_scale"
                || name.starts_with("continuation_target_")
            {
                vec![1.0; count]
            } else if name == "initial_deter" {
                (0..count).map(|i| (i as f32 * 0.13).sin() * 0.2).collect()
            } else {
                let width = *node.ty.shape.last().unwrap();
                let signed = if name.starts_with("initial_action_target_") && time % 2 == 1 {
                    -0.8
                } else {
                    1.0
                };
                (0..count)
                    .map(|i| {
                        if i % width == (i / width + time) % width {
                            signed
                        } else {
                            0.0
                        }
                    })
                    .collect()
            };
            session.set_input(name, &values);
        }
        let names = session
            .param_names()
            .iter()
            .filter(|n| n.starts_with("encoder.") || n.starts_with("behavior."))
            .map(|n| n.to_string())
            .collect::<Vec<_>>();
        let refs = names.iter().map(String::as_str).collect::<Vec<_>>();
        let before = session.read_params(&refs);
        session.clear_optimizer();
        session.step();
        session.wait();
        check_device();
        let mut nonzero = 0;
        for name in &names {
            if !name.starts_with("encoder.") {
                assert!(!session.has_param_grad(name), "unfrozen behavior {name}");
                continue;
            }
            let mut gradient = vec![0.0; session.param_size(name).unwrap()];
            session.read_param_grad(name, &mut gradient);
            assert!(gradient.iter().all(|x| x.is_finite()), "{name}");
            assert!(
                gradient.iter().any(|&x| x != 0.0),
                "zero policy gradient {name}"
            );
            nonzero += 1;
        }
        assert_eq!(nonzero, 148);
        configure_d3_optimizer(&mut session, &config, 0, config.learning_rate);
        session.step();
        session.wait();
        check_device();
        let after = session.read_params(&refs);
        let mut changed = 0;
        for ((name, before), after) in names.iter().zip(before).zip(after) {
            assert!(after.iter().all(|x| x.is_finite()), "{name}");
            if name.starts_with("encoder.") {
                changed += usize::from(before != after);
            } else {
                assert_eq!(before, after, "behavior head was updated {name}");
            }
        }
        assert!(changed > 0);
        eprintln!(
            "isolated_policy_tiny: {}",
            serde_json::json!({"encoder_nonzero_gradients": nonzero, "encoder_changed_tensors": changed, "behavior_heads_unchanged": true})
        );
    }

    #[test]
    fn tiny_graphs_expose_d3_shapes() {
        let config = DreamerConfig::tiny(3);
        let size = config.network();
        let training = build_training_graph(&config, config.world_backprop_length);
        assert_eq!(training.outputs().len(), 9);
        assert!(!training.nodes().iter().any(|n| matches!(
            &n.op, meganeura::graph::Op::Parameter { name } if name.starts_with("behavior.actor.")
        )));
        let grouped_weight = training
            .nodes()
            .iter()
            .find_map(|node| match &node.op {
                meganeura::graph::Op::Parameter { name }
                    if name == "world.dynamics.core.dynhid0.weight" =>
                {
                    Some(node.ty.shape.as_slice())
                }
                _ => None,
            })
            .expect("grouped RSSM weight");
        assert_eq!(
            grouped_weight,
            [
                size.blocks,
                size.deter / size.blocks + 3 * size.hidden,
                size.deter / size.blocks,
            ]
        );
        let observe = build_observe_graph(&config, 2);
        assert_eq!(
            observe.node(observe.outputs()[0]).ty.shape,
            vec![2, size.deter]
        );
        assert_eq!(
            observe.node(observe.outputs()[1]).ty.shape,
            vec![2 * size.stoch, size.classes]
        );
        assert_eq!(
            observe.node(observe.outputs()[3]).ty.shape,
            vec![2, OBSERVATION_GRID * OBSERVATION_GRID * size.vision_depth]
        );
        let transition = build_transition_graph(&config, 5);
        assert_eq!(
            transition.node(transition.outputs()[0]).ty.shape,
            vec![5, size.deter]
        );
        let heads = build_head_graph(&config, 5);
        assert_eq!(
            heads.node(heads.outputs()[0]).ty.shape,
            vec![5, config.value_bins]
        );
        assert_eq!(heads.node(heads.outputs()[1]).ty.shape, vec![5, 1]);
        assert_eq!(heads.outputs().len(), 2);
        let imagination = build_imagination_head_graph(&config, 5);
        assert_eq!(imagination.outputs().len(), 3);
        assert_eq!(
            imagination.node(imagination.outputs()[2]).ty.shape,
            vec![5, config.feature_dim()]
        );
        let decoder = build_observation_prediction_graph(&config, 5);
        assert_eq!(
            decoder.node(decoder.outputs()[0]).ty.shape,
            vec![
                5 * OBSERVATION_GRID * OBSERVATION_GRID,
                OBSERVATION_CHANNELS
            ]
        );
    }

    #[test]
    fn prediction_only_omits_decoder_parameters() {
        let mut config = DreamerConfig::tiny(3);
        config.loss_scales.reconstruction = 0.0;
        config.loss_scales.future_prediction = 0.25;
        let graph = build_training_graph(&config, config.world_backprop_length);
        let names = graph
            .nodes()
            .iter()
            .filter_map(|node| match &node.op {
                meganeura::graph::Op::Parameter { name } => Some(name.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(!names.iter().any(|name| name.starts_with("world.decoder.")));
        assert!(
            names
                .iter()
                .any(|name| name.starts_with("world.future_predictor."))
        );
    }

    #[test]
    fn future_and_reconstruction_heads_share_spatial_parameterization() {
        let mut config = DreamerConfig::tiny(3);
        config.loss_scales.future_prediction = 0.25;
        let graph = build_training_graph(&config, config.world_backprop_length);
        let parameters = graph
            .nodes()
            .iter()
            .filter_map(|node| match &node.op {
                meganeura::graph::Op::Parameter { name } => Some((name, &node.ty.shape)),
                _ => None,
            })
            .collect::<std::collections::HashMap<_, _>>();
        for (name, shape) in &parameters {
            if let Some(suffix) = name.strip_prefix("world.decoder.") {
                let future = parameters[&format!("world.future_predictor.{suffix}")];
                if suffix == "trunk.weight" {
                    assert_eq!(future[0], config.network().deter);
                    assert_eq!(shape[0], config.feature_dim());
                    assert_eq!(future[1..], shape[1..]);
                } else {
                    assert_eq!(future, *shape);
                }
            }
        }
    }

    #[test]
    fn tiny_world_training_graph_autodiffs_and_compiles() {
        let config = DreamerConfig::tiny(3);
        let graph = build_training_graph(&config, config.world_backprop_length);
        let (plan, _) = meganeura::compile_training_graph(&graph);
        assert!(!plan.dispatches.is_empty());
    }

    #[test]
    #[ignore = "checks a composed continuation loss beyond one GPU workgroup"]
    fn large_composed_continuation_loss_covers_every_row() {
        use super::super::runtime::build_session;
        use meganeura::Mode;
        use std::sync::Arc;

        let gpu = Arc::new(crate::init_gpu_context().unwrap());
        for rows in [513, 1024] {
            let mut graph = Graph::new();
            let logits = graph.parameter("logits", &[rows, 1]);
            let prediction = graph.sigmoid(logits);
            let target = graph.input("target", &[rows, 1]);
            let loss = mean_binary_cross_entropy(&mut graph, prediction, target, rows);
            let loss = scale(&mut graph, loss, 0.3);
            graph.set_outputs(vec![loss]);
            let mut session = build_session(&graph, &gpu, Mode::Training, false);
            let logits = (0..rows)
                .map(|row| (row as f32 * 0.17).sin() * 2.0)
                .collect::<Vec<_>>();
            let target = (0..rows)
                .map(|row| (row % 7) as f32 / 6.0)
                .collect::<Vec<_>>();
            session.set_parameter("logits", &logits);
            session.set_input("target", &target);
            session.step();
            session.wait();
            let mut actual_loss = [0.0];
            let mut actual_gradient = vec![0.0; rows];
            session.read_output_by_index(0, &mut actual_loss);
            session.read_param_grad("logits", &mut actual_gradient);
            let mut expected_loss = 0.0_f64;
            for ((logit, target), actual) in logits.iter().zip(&target).zip(&actual_gradient) {
                let p = 1.0 / (1.0 + (-f64::from(*logit)).exp());
                let t = f64::from(*target);
                expected_loss -= 0.3 * (t * p.ln() + (1.0 - t) * (1.0 - p).ln()) / rows as f64;
                let expected_gradient = 0.3 * (p - t) / rows as f64;
                assert!((f64::from(*actual) - expected_gradient).abs() < 1e-8);
            }
            assert!((f64::from(actual_loss[0]) - expected_loss).abs() < 1e-6);
        }
    }

    #[test]
    #[ignore = "requires a separately declared driver-bound production-gradient GPU diagnostic"]
    fn temporal_batching_matches_serial_losses_and_gradients() {
        use super::super::runtime::{build_session, initialize_d3};
        use meganeura::{Mode, Session};
        use std::sync::Arc;

        let driver = gradient_driver(
            std::env::var("KINDLE_GRADIENT_DIAGNOSTIC").ok().as_deref(),
            std::env::var("KINDLE_INIT_EXPECTED_DRIVER").ok().as_deref(),
            std::env::var("KINDLE_FULL_WORLD_PARITY").ok().as_deref(),
        );

        fn fill_inputs(session: &mut Session, config: &DreamerConfig) {
            let batch = config.batch_size;
            let size = config.network();
            let signal = |count: usize, time: usize| {
                (0..count)
                    .map(|index| ((index * 17 + time * 37) as f32 * 0.013).sin() * 0.3)
                    .collect::<Vec<_>>()
            };
            let one_hot = |rows: usize, width: usize, offset: usize| {
                (0..rows * width)
                    .map(|index| f32::from(index % width == (index / width + offset) % width))
                    .collect::<Vec<_>>()
            };
            session.set_input("initial_deter", &signal(batch * size.deter, 7));
            session.set_input(
                "initial_stoch",
                &one_hot(batch * size.stoch, size.classes, 1),
            );
            for time in 0..config.batch_length {
                for (name, width) in [
                    ("keep_deter", size.deter),
                    ("keep_stoch", size.stoch * size.classes),
                    ("keep_action", config.action_count),
                ] {
                    let keep = (0..batch * width)
                        .map(|index| f32::from(time != index / width))
                        .collect::<Vec<_>>();
                    session.set_input(&format!("{name}_{time}"), &keep);
                }
                session.set_input(
                    &format!("observation_{time}"),
                    &signal(batch * config.observation_dim(), time),
                );
                session.set_input(
                    &format!("previous_action_{time}"),
                    &one_hot(batch, config.action_count, time),
                );
                session.set_input(
                    &format!("posterior_sample_{time}"),
                    &one_hot(batch * size.stoch, size.classes, time + 2),
                );
                for (name, offset) in [
                    ("reward_target", 2 * time),
                    ("replay_value_target", time + 1),
                    ("replay_slow_target", time + 3),
                ] {
                    session.set_input(
                        &format!("{name}_{time}"),
                        &one_hot(batch, config.value_bins, offset),
                    );
                }
                session.set_input(
                    &format!("continuation_target_{time}"),
                    &(0..batch)
                        .map(|row| {
                            f32::from((time + row) % 3 != 0) * config.continuation_discount()
                        })
                        .collect::<Vec<_>>(),
                );
                session.set_input(
                    &format!("replay_value_weight_{time}"),
                    &(0..batch)
                        .map(|row| {
                            if time + 1 == config.batch_length {
                                0.0
                            } else {
                                (row + 1) as f32 / batch as f32
                            }
                        })
                        .collect::<Vec<_>>(),
                );
            }
        }

        // Opt into the production B16/T64 graph on a GPU with enough memory
        // for both reference and grouped sessions. Larger matmuls can select
        // different forward kernels; derivative operands remain F32.
        let full = std::env::var_os("KINDLE_FULL_WORLD_PARITY").is_some();
        let configs = if full {
            let mut config = DreamerConfig::new(18);
            config.world_backprop_length = config.batch_length;
            config.loss_scales.reconstruction = 0.0;
            config.loss_scales.future_prediction = 0.25;
            vec![config]
        } else {
            [(3, 0.0, true, 1.0), (4, 0.75, false, 0.0)]
                .map(
                    |(length, reconstruction, replay_value_gradient, free_nats)| {
                        let mut config = DreamerConfig::tiny(3);
                        config.batch_size = 3;
                        config.batch_length = length;
                        config.world_backprop_length = length;
                        config.loss_scales.future_prediction = 0.25;
                        config.loss_scales.reconstruction = reconstruction;
                        config.replay_value_gradient = replay_value_gradient;
                        config.free_nats = free_nats;
                        config
                    },
                )
                .to_vec()
        };
        let loss_tolerance = if full { 3e-4 } else { 3e-5 };
        let gradient_tolerance = if full { 3e-3 } else { 3e-4 };
        gradient_mark("device.before", 0, serde_json::json!({}));
        let gpu = Arc::new(crate::init_gpu_context().unwrap());
        let device = crate::gpu_device_info(gpu.device_information());
        gradient_device(&device, driver);
        gradient_mark("device.ready", 0, serde_json::to_value(&device).unwrap());
        for config in configs {
            let length = config.batch_length;
            gradient_mark("config", 0, serde_json::to_value(&config).unwrap());
            let mut sessions = [1, length].map(|group| {
                gradient_mark("graph.before", group, serde_json::json!({}));
                let graph = build_training_graph_grouped(&config, length, group);
                gradient_mark("graph.ready", group, serde_json::json!({}));
                let mut session = build_session(&graph, &gpu, Mode::Training, false);
                gradient_mark("session.ready", group, serde_json::json!({}));
                initialize_d3(&mut session, &graph, config.seed);
                gradient_mark("parameters.initialized", group, serde_json::json!({}));
                // Exercise the input gradients of heads that D3 initializes to zero.
                for name in ["world.reward.out.weight", "behavior.value.out.weight"] {
                    let values = (0..session.param_size(name).unwrap())
                        .map(|index| (index as f32 * 0.17).sin() * 0.02)
                        .collect::<Vec<_>>();
                    session.set_parameter(name, &values);
                }
                fill_inputs(&mut session, &config);
                gradient_mark("inputs.ready", group, serde_json::json!({}));
                session.clear_optimizer();
                gradient_mark("optimizer.cleared", group, serde_json::json!({}));
                session.step();
                gradient_mark("step.submitted", group, serde_json::json!({}));
                session.wait();
                gradient_mark("step.wait_returned", group, serde_json::json!({}));
                session
            });
            gradient_mark("comparisons.before", 0, serde_json::json!({}));
            let [serial, grouped] = &mut sessions;
            for metric in 0..9 {
                let mut left = [0.0];
                let mut right = [0.0];
                serial.read_output_by_index(metric, &mut left);
                grouped.read_output_by_index(metric, &mut right);
                let tolerance = loss_tolerance * left[0].abs().max(1.0);
                gradient_mark(
                    "metric",
                    0,
                    serde_json::json!({"index": metric, "serial": left[0], "grouped": right[0]}),
                );
                assert!(
                    (left[0] - right[0]).abs() <= tolerance,
                    "T={length}, metric {metric}: {left:?} vs {right:?}"
                );
            }
            let mut names = serial.param_names();
            let mut grouped_names = grouped.param_names();
            names.sort_unstable();
            grouped_names.sort_unstable();
            assert_eq!(names, grouped_names);
            let parameter_count = names.len();
            let mut compared_gradients = 0;
            let mut nonzero_gradients = 0;
            let mut worst_relative = 0.0_f64;
            for name in names {
                assert_eq!(serial.param_size(name), grouped.param_size(name));
                assert_eq!(serial.has_param_grad(name), grouped.has_param_grad(name));
                gradient_mark(
                    "parameter",
                    0,
                    serde_json::json!({
                        "name": name, "elements": serial.param_size(name).unwrap(),
                        "has_gradient": serial.has_param_grad(name),
                    }),
                );
                if name.starts_with("behavior.") {
                    assert!(!serial.has_param_grad(name), "replay critic remains frozen");
                }
                if !serial.has_param_grad(name) {
                    continue;
                }
                let mut left = vec![0.0; serial.param_size(name).unwrap()];
                let mut right = vec![0.0; left.len()];
                serial.read_param_grad(name, &mut left);
                grouped.read_param_grad(name, &mut right);
                assert!(left.iter().chain(&right).all(|value| value.is_finite()));
                let squared = |values: &[f32]| {
                    values
                        .iter()
                        .map(|value| f64::from(*value).powi(2))
                        .sum::<f64>()
                };
                let difference = left
                    .iter()
                    .zip(&right)
                    .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let norm = squared(&left).max(squared(&right)).sqrt();
                let relative = difference / norm.max(1e-12);
                compared_gradients += 1;
                nonzero_gradients += usize::from(norm > 0.0);
                gradient_mark(
                    "gradient",
                    0,
                    serde_json::json!({
                        "name": name, "norm": norm, "difference": difference, "relative": relative,
                    }),
                );
                worst_relative = worst_relative.max(relative);
                assert!(
                    difference <= gradient_tolerance * norm + 1e-7,
                    "T={length}, {name}: gradient relative L2={relative}, norm={norm}"
                );
            }
            eprintln!("T={length}, worst per-parameter gradient relative L2={worst_relative}");
            assert!(
                nonzero_gradients > 0,
                "all-zero gradients do not establish execution"
            );
            gradient_mark(
                "complete",
                0,
                serde_json::json!({
                    "metrics": 9, "parameters": parameter_count,
                    "compared_gradients": compared_gradients,
                    "nonzero_gradients": nonzero_gradients, "worst_relative": worst_relative,
                }),
            );
        }
    }

    fn gradient_driver(
        selection: Option<&str>,
        driver: Option<&str>,
        full: Option<&str>,
    ) -> &'static str {
        assert_eq!(selection, Some("production-world-driver-20260916"));
        assert_eq!(full, Some("1"), "production B16/T64 is required");
        match driver {
            Some("580.178.04") => "580.178.04",
            Some("595.91.07") => "595.91.07",
            _ => panic!("an explicit listed driver is required"),
        }
    }

    fn gradient_device(info: &crate::GpuDeviceInfo, driver: &str) {
        assert_eq!(info.device_name, "NVIDIA GeForce RTX 5080");
        assert_eq!(info.driver_name, "NVIDIA");
        assert_eq!(info.driver_info, driver);
        assert_eq!(info.requested_device_id.as_deref(), Some("0x2c02"));
        assert!(!info.is_software_emulated);
    }

    fn gradient_mark(phase: &str, group: usize, data: serde_json::Value) {
        use std::io::Write;
        let mut writer = std::io::stderr().lock();
        serde_json::to_writer(&mut writer, &serde_json::json!({
            "kindle_gradient": 1, "pid": std::process::id(), "phase": phase, "group": group,
            "unix_ns": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos(),
            "data": data,
        })).unwrap();
        writer.write_all(b"\n").unwrap();
        writer.flush().unwrap();
    }

    #[test]
    fn gradient_declaration_requires_full_explicit_driver() {
        let selection = Some("production-world-driver-20260916");
        for driver in ["580.178.04", "595.91.07"] {
            assert_eq!(gradient_driver(selection, Some(driver), Some("1")), driver);
        }
        for (tag, driver, full) in [
            (None, Some("580.178.04"), Some("1")),
            (Some("other"), Some("580.178.04"), Some("1")),
            (selection, None, Some("1")),
            (selection, Some("595.71.05"), Some("1")),
            (selection, Some("580.178.04"), None),
            (selection, Some("580.178.04"), Some("0")),
        ] {
            assert!(std::panic::catch_unwind(|| gradient_driver(tag, driver, full)).is_err());
        }
    }

    #[test]
    fn gradient_device_rejects_mismatches() {
        let good = crate::GpuDeviceInfo {
            device_name: "NVIDIA GeForce RTX 5080".into(),
            driver_name: "NVIDIA".into(),
            driver_info: "580.178.04".into(),
            requested_device_id: Some("0x2c02".into()),
            is_software_emulated: false,
        };
        gradient_device(&good, "580.178.04");
        for field in 0..5 {
            let mut wrong = good.clone();
            match field {
                0 => wrong.device_name = "other".into(),
                1 => wrong.driver_name = "other".into(),
                2 => wrong.driver_info = "595.91.07".into(),
                3 => wrong.requested_device_id = None,
                _ => wrong.is_software_emulated = true,
            }
            assert!(std::panic::catch_unwind(|| gradient_device(&wrong, "580.178.04")).is_err());
        }
    }
}
