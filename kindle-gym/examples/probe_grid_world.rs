use std::path::{Path, PathBuf};

use clap::Parser;
use kindle::vision::{DinoPerception, OBSERVATION_CHANNELS};
use kindle::{ActionMode, DreamerAgent, Environment};
use kindle_gym::grid_world::{
    ACTION_COUNT, GridWorld, HEIGHT, POSITION_RANDOMIZATION_SEED_XOR, WIDTH,
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde_json::json;

#[derive(Debug, Parser)]
#[command(about = "Probe GridWorld state separability before or after Dreamer's RSSM")]
struct Arguments {
    /// DINOv3 ViT-S/16 model.safetensors.
    dino_checkpoint: PathBuf,

    /// Probe the trained posterior feature from this Dreamer checkpoint.
    /// Without this option, probe the frozen DINO observation directly.
    #[arg(long)]
    checkpoint: Option<PathBuf>,

    /// Number of frames from a random valid-action trajectory.
    #[arg(long, default_value_t = 500)]
    samples: usize,

    /// Maximum open-loop prior horizon scored from each posterior state.
    #[arg(long, default_value_t = 15)]
    rollout_horizon: usize,

    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Teleport to a uniformly random cell after each action so position must
    /// be inferred from the rendered observation rather than action history.
    #[arg(long)]
    randomize_position: bool,

    /// Sample all four actions, including border no-ops, instead of applying
    /// GridWorld's action mask. This matches unmasked training runs.
    #[arg(long)]
    all_actions: bool,
}

struct Sample {
    representations: Vec<Vec<f32>>,
    position: usize,
    dense_right: usize,
    food: usize,
    rewarded: usize,
    reward_prediction: Option<f32>,
    posterior_observation_mse: Option<f32>,
    next_rewarded: Option<usize>,
    prior_reward_prediction: Option<f32>,
    prior_reward_rollout: Vec<(f32, usize)>,
    prior_observation_rollout: Vec<ObservationPredictionError>,
    invalid_action_probability: Option<f32>,
    unmasked_argmax_invalid: Option<usize>,
    policy_reward_alignment: Option<PolicyRewardAlignment>,
}

struct SampleCollection {
    samples: Vec<Sample>,
    dino_patch_statistics: DinoPatchStatistics,
    observation_decoder_depth: Option<usize>,
}

/// Sufficient statistics for the best affine low-rank approximation of the
/// frozen DINO patch vectors. The decoder's final shared linear layer maps
/// `vision_depth` hidden channels to 64 DINO channels, so this is a hard lower
/// bound on its reconstruction MSE independent of the RSSM or optimizer.
struct DinoPatchStatistics {
    count: usize,
    sum: [f64; OBSERVATION_CHANNELS],
    second_moment: Vec<f64>,
}

impl DinoPatchStatistics {
    fn new() -> Self {
        Self {
            count: 0,
            sum: [0.0; OBSERVATION_CHANNELS],
            second_moment: vec![0.0; OBSERVATION_CHANNELS * OBSERVATION_CHANNELS],
        }
    }

    fn add_observation(&mut self, observation: &[f32]) {
        let (patches, remainder) = observation.as_chunks::<OBSERVATION_CHANNELS>();
        assert!(remainder.is_empty());
        for patch in patches {
            self.count += 1;
            for (channel, value) in patch.iter().enumerate() {
                let value = f64::from(*value);
                self.sum[channel] += value;
                for (other_channel, other) in patch.iter().enumerate() {
                    self.second_moment[channel * OBSERVATION_CHANNELS + other_channel] +=
                        value * f64::from(*other);
                }
            }
        }
    }

    fn metrics(&self) -> serde_json::Value {
        assert!(self.count > 0);
        let count = self.count as f64;
        let mean = self.sum.map(|sum| sum / count);
        let mut covariance = self
            .second_moment
            .iter()
            .map(|moment| moment / count)
            .collect::<Vec<_>>();
        for row in 0..OBSERVATION_CHANNELS {
            for column in 0..OBSERVATION_CHANNELS {
                covariance[row * OBSERVATION_CHANNELS + column] -= mean[row] * mean[column];
            }
        }
        let mut eigenvalues = symmetric_eigenvalues(covariance, OBSERVATION_CHANNELS);
        eigenvalues.sort_by(|left, right| right.total_cmp(left));
        for value in &mut eigenvalues {
            *value = value.max(0.0);
        }
        let total_variance = eigenvalues.iter().sum::<f64>();
        let rank_floors = [0, 4, 16, 24, 32, 48, 64]
            .into_iter()
            .map(|rank| {
                let retained_variance = eigenvalues[..rank].iter().sum::<f64>();
                let residual_variance = (total_variance - retained_variance).max(0.0);
                json!({
                    "rank": rank,
                    "minimum_mean_squared_error": residual_variance
                        / OBSERVATION_CHANNELS as f64,
                    "explained_variance": (total_variance > 0.0)
                        .then(|| retained_variance / total_variance),
                })
            })
            .collect::<Vec<_>>();
        json!({
            "patch_samples": self.count,
            "channel_count": OBSERVATION_CHANNELS,
            "rank_floors": rank_floors,
        })
    }
}

/// Eigenvalues of a real symmetric row-major matrix via cyclic Jacobi
/// rotations. This keeps the standalone probe dependency-free; 64 channels
/// make the small O(n^3) diagnostic negligible next to DINO inference.
fn symmetric_eigenvalues(mut matrix: Vec<f64>, size: usize) -> Vec<f64> {
    assert_eq!(matrix.len(), size * size);
    for _ in 0..64 {
        let mut largest_off_diagonal = 0.0_f64;
        for row in 0..size {
            for column in row + 1..size {
                let offset = row * size + column;
                let off_diagonal = matrix[offset];
                largest_off_diagonal = largest_off_diagonal.max(off_diagonal.abs());
                if off_diagonal.abs() <= 1e-14 {
                    continue;
                }
                let row_diagonal = matrix[row * size + row];
                let column_diagonal = matrix[column * size + column];
                let tau = (column_diagonal - row_diagonal) / (2.0 * off_diagonal);
                let sign = if tau >= 0.0 { 1.0 } else { -1.0 };
                let tangent = sign / (tau.abs() + (1.0 + tau * tau).sqrt());
                let cosine = 1.0 / (1.0 + tangent * tangent).sqrt();
                let sine = tangent * cosine;

                for index in 0..size {
                    if index == row || index == column {
                        continue;
                    }
                    let index_row = matrix[index * size + row];
                    let index_column = matrix[index * size + column];
                    let rotated_row = cosine * index_row - sine * index_column;
                    let rotated_column = sine * index_row + cosine * index_column;
                    matrix[index * size + row] = rotated_row;
                    matrix[row * size + index] = rotated_row;
                    matrix[index * size + column] = rotated_column;
                    matrix[column * size + index] = rotated_column;
                }
                matrix[row * size + row] = row_diagonal - tangent * off_diagonal;
                matrix[column * size + column] = column_diagonal + tangent * off_diagonal;
                matrix[offset] = 0.0;
                matrix[column * size + row] = 0.0;
            }
        }
        if largest_off_diagonal <= 1e-12 {
            break;
        }
    }
    (0..size)
        .map(|index| matrix[index * size + index])
        .collect()
}

#[derive(Clone, Copy)]
struct ObservationPredictionError {
    model_mse: f32,
    persistence_mse: f32,
}

struct PendingObservationRollout {
    sample_index: usize,
    next_horizon: usize,
    predictions: Vec<Vec<f32>>,
    persistence: Vec<f32>,
}

#[derive(Clone, Copy)]
struct PolicyRewardAlignment {
    policy_expected_prior_reward: f32,
    uniform_expected_prior_reward: f32,
    policy_argmax_prior_reward: f32,
    best_prior_reward: f32,
    argmax_agreement: usize,
    has_rewarding_action: usize,
    rewarding_action_probability: f32,
    uniform_rewarding_action_probability: f32,
    policy_argmax_rewarded: usize,
    prior_argmax_rewarded: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = Arguments::parse();
    if arguments.samples < 100 {
        return Err("--samples must be at least 100".into());
    }
    if arguments.rollout_horizon == 0 || arguments.rollout_horizon > 64 {
        return Err("--rollout-horizon must be in 1..=64".into());
    }

    let (source, collection, representation_names) = if let Some(checkpoint) = &arguments.checkpoint
    {
        (
            "dreamer_posterior",
            collect_latent_samples(&arguments, checkpoint)?,
            ["encoder", "full", "deter", "stoch"].as_slice(),
        )
    } else {
        (
            "dinov3_observation",
            collect_perception_samples(&arguments)?,
            ["full"].as_slice(),
        )
    };

    let SampleCollection {
        samples,
        dino_patch_statistics,
        observation_decoder_depth,
    } = collection;
    let dino_patch_affine_rank_floor = dino_patch_statistics.metrics();
    let mut representations = serde_json::Map::new();
    let mut feature_dims = serde_json::Map::new();
    for (index, name) in representation_names.iter().enumerate() {
        representations.insert((*name).into(), label_probes(&samples, index));
        feature_dims.insert(
            (*name).into(),
            json!(samples[0].representations[index].len()),
        );
    }
    let reward_head = samples[0].reward_prediction.map(|_| {
        json!({
            "dense_right": reward_prediction_metrics(&samples, |sample| sample.dense_right),
            "previous_sparse_reward": reward_prediction_metrics(
                &samples,
                |sample| sample.rewarded,
            ),
        })
    });
    let posterior_dino_reconstruction = posterior_observation_prediction_metrics(&samples);
    let one_step_prior_reward = samples
        .iter()
        .any(|sample| sample.prior_reward_prediction.is_some())
        .then(|| prior_reward_prediction_metrics(&samples));
    let open_loop_prior_reward = (0..arguments.rollout_horizon)
        .filter_map(|horizon| {
            rollout_reward_prediction_metrics(&samples, horizon).map(|metrics| {
                json!({
                    "horizon": horizon + 1,
                    "metrics": metrics,
                })
            })
        })
        .collect::<Vec<_>>();
    let open_loop_dino_reconstruction = (0..arguments.rollout_horizon)
        .filter_map(|horizon| {
            rollout_observation_prediction_metrics(&samples, horizon).map(|metrics| {
                json!({
                    "horizon": horizon + 1,
                    "metrics": metrics,
                })
            })
        })
        .collect::<Vec<_>>();
    let posterior_policy_action_mask = policy_action_mask_metrics(&samples);
    let posterior_policy_reward_alignment = policy_reward_alignment_metrics(&samples);
    println!(
        "{}",
        serde_json::to_string(&json!({
            "event": "representation_probe",
            "source": source,
            "samples": arguments.samples,
            "seed": arguments.seed,
            "randomize_position": arguments.randomize_position,
            "all_actions": arguments.all_actions,
            "feature_dims": feature_dims,
            "observation_decoder_depth": observation_decoder_depth,
            "representations": representations,
            "reward_head": reward_head,
            "dino_patch_affine_rank_floor": dino_patch_affine_rank_floor,
            "posterior_dino_reconstruction": posterior_dino_reconstruction,
            "one_step_prior_reward": one_step_prior_reward,
            "open_loop_prior_reward": open_loop_prior_reward,
            "open_loop_dino_reconstruction": open_loop_dino_reconstruction,
            "posterior_policy_action_mask": posterior_policy_action_mask,
            "posterior_policy_reward_alignment": posterior_policy_reward_alignment,
        }))?
    );
    Ok(())
}

fn collect_perception_samples(
    arguments: &Arguments,
) -> Result<SampleCollection, Box<dyn std::error::Error>> {
    let mut perception = DinoPerception::load_vits16(&arguments.dino_checkpoint, None, None)?;
    let mut environment = GridWorld::new();
    let mut rng = StdRng::seed_from_u64(arguments.seed);
    let mut position_rng = StdRng::seed_from_u64(arguments.seed ^ POSITION_RANDOMIZATION_SEED_XOR);
    environment.reset();
    let mut samples = Vec::with_capacity(arguments.samples);
    let mut dino_patch_statistics = DinoPatchStatistics::new();
    let mut rewarded = false;

    while samples.len() < arguments.samples {
        let frame = environment.render();
        let (x, y) = environment.position();
        let observation =
            perception.encode_frame_rgb8(frame.pixels(), frame.width(), frame.height());
        dino_patch_statistics.add_observation(observation.as_slice());
        samples.push(Sample {
            representations: vec![observation.as_slice().to_vec()],
            position: y * WIDTH + x,
            dense_right: usize::from(x >= WIDTH - 2),
            food: environment.food_index(),
            rewarded: usize::from(rewarded),
            reward_prediction: None,
            posterior_observation_mse: None,
            next_rewarded: None,
            prior_reward_prediction: None,
            prior_reward_rollout: Vec::new(),
            prior_observation_rollout: Vec::new(),
            invalid_action_probability: None,
            unmasked_argmax_invalid: None,
            policy_reward_alignment: None,
        });
        if samples.len() == arguments.samples {
            break;
        }

        let action = random_action(&environment, arguments.all_actions, &mut rng);
        let mut transition = environment.step(action);
        if arguments.randomize_position {
            environment.randomize_position(&mut transition, &mut position_rng);
        }
        rewarded = transition.reward.extrinsic != 0.0;
        if transition.terminated || transition.truncated {
            environment.reset();
            rewarded = false;
        }
    }
    Ok(SampleCollection {
        samples,
        dino_patch_statistics,
        observation_decoder_depth: None,
    })
}

fn collect_latent_samples(
    arguments: &Arguments,
    checkpoint: &Path,
) -> Result<SampleCollection, Box<dyn std::error::Error>> {
    let mut agent = DreamerAgent::restore(checkpoint, &arguments.dino_checkpoint, None)?;
    let observation_decoder_depth = agent.core().config().observation_decoder_depth();
    let mut environment = GridWorld::new();
    let mut rng = StdRng::seed_from_u64(arguments.seed);
    let mut position_rng = StdRng::seed_from_u64(arguments.seed ^ POSITION_RANDOMIZATION_SEED_XOR);
    let first = environment.reset();
    agent.begin_episode(&first);
    let mut samples = Vec::with_capacity(arguments.samples);
    let mut dino_patch_statistics = DinoPatchStatistics::new();
    let mut pending_observations = Vec::<PendingObservationRollout>::new();
    let mut rewarded = false;

    while samples.len() < arguments.samples {
        dino_patch_statistics.add_observation(agent.dino_observation());
        score_pending_observations(
            &mut pending_observations,
            &mut samples,
            agent.dino_observation(),
        );
        push_latent_sample(
            &mut samples,
            &mut agent,
            &environment,
            rewarded,
            arguments.all_actions,
        );
        if samples.len() == arguments.samples {
            break;
        }
        let mut preview_environment = environment.clone();
        let mut preview_rng = rng.clone();
        let mut preview_position_rng = position_rng.clone();
        let mut rollout_actions = Vec::with_capacity(arguments.rollout_horizon);
        let mut rollout_labels = Vec::with_capacity(arguments.rollout_horizon);
        for _ in 0..arguments.rollout_horizon {
            let action = random_action(
                &preview_environment,
                arguments.all_actions,
                &mut preview_rng,
            );
            let mut transition = preview_environment.step(action);
            if arguments.randomize_position {
                preview_environment.randomize_position(&mut transition, &mut preview_position_rng);
            }
            rollout_actions.push(action);
            rollout_labels.push(usize::from(transition.reward.extrinsic != 0.0));
            if transition.terminated || transition.truncated {
                break;
            }
        }
        let (rollout_predictions, rollout_observations) =
            agent.prior_diagnostic_rollout(&rollout_actions);
        assert_eq!(rollout_predictions.len(), rollout_observations.len());
        let sample_index = samples.len() - 1;
        let sample = samples.last_mut().expect("current latent sample");
        sample.prior_reward_prediction = Some(rollout_predictions[0]);
        sample.next_rewarded = Some(rollout_labels[0]);
        sample.prior_reward_rollout = rollout_predictions
            .into_iter()
            .zip(rollout_labels)
            .collect();
        pending_observations.push(PendingObservationRollout {
            sample_index,
            next_horizon: 0,
            predictions: rollout_observations,
            persistence: agent.dino_observation().to_vec(),
        });

        let action = random_action(&environment, arguments.all_actions, &mut rng);
        assert_eq!(action, rollout_actions[0]);
        let mut forced_mask = [false; ACTION_COUNT];
        forced_mask[action] = true;
        assert_eq!(agent.act(ActionMode::Sample, Some(&forced_mask)), action);
        let mut transition = environment.step(action);
        if arguments.randomize_position {
            environment.randomize_position(&mut transition, &mut position_rng);
        }
        rewarded = transition.reward.extrinsic != 0.0;
        assert_eq!(usize::from(rewarded), sample.next_rewarded.unwrap());
        let ended = transition.terminated || transition.truncated;
        agent.observe(&transition);
        if ended {
            pending_observations.clear();
            agent.begin_episode(&environment.reset());
            rewarded = false;
        }
    }
    Ok(SampleCollection {
        samples,
        dino_patch_statistics,
        observation_decoder_depth: Some(observation_decoder_depth),
    })
}

fn push_latent_sample(
    samples: &mut Vec<Sample>,
    agent: &mut DreamerAgent,
    environment: &GridWorld,
    rewarded: bool,
    all_actions: bool,
) {
    let (x, y) = environment.position();
    let deter_dim = agent.core().config().network().deter;
    let latent = agent.latent_feature();
    let representations = vec![
        agent.encoded_observation().to_vec(),
        latent.to_vec(),
        latent[..deter_dim].to_vec(),
        latent[deter_dim..].to_vec(),
    ];
    let reward_prediction = agent.posterior_reward_prediction();
    let observation = agent.dino_observation().to_vec();
    let posterior_observation = agent.posterior_observation_prediction();
    let posterior_observation_mse = mean_squared_error(&posterior_observation, &observation);
    let probabilities = agent.posterior_action_probabilities(None);
    let action_mask = environment.action_mask().expect("GridWorld action mask");
    let invalid_action_probability = probabilities
        .iter()
        .zip(&action_mask)
        .filter_map(|(probability, valid)| (!valid).then_some(*probability))
        .sum();
    let unmasked_argmax = probabilities
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(action, _)| action)
        .expect("non-empty action distribution");
    let policy_probabilities = if all_actions {
        probabilities.clone()
    } else {
        agent.posterior_action_probabilities(Some(&action_mask))
    };
    let eligible_actions = (0..ACTION_COUNT)
        .filter(|action| all_actions || action_mask[*action])
        .collect::<Vec<_>>();
    let prior_rewards = eligible_actions
        .iter()
        .map(|action| agent.prior_reward_prediction(*action))
        .collect::<Vec<_>>();
    let actual_rewards = eligible_actions
        .iter()
        .map(|action| {
            let mut preview = environment.clone();
            usize::from(preview.step(*action).reward.extrinsic != 0.0)
        })
        .collect::<Vec<_>>();
    let policy_argmax_index = eligible_actions
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            policy_probabilities[**left].total_cmp(&policy_probabilities[**right])
        })
        .map(|(index, _)| index)
        .expect("at least one eligible action");
    let prior_argmax_index = prior_rewards
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .expect("at least one eligible prior prediction");
    let rewarding_action_count = actual_rewards.iter().sum::<usize>();
    let policy_expected_prior_reward = eligible_actions
        .iter()
        .zip(&prior_rewards)
        .map(|(action, reward)| policy_probabilities[*action] * reward)
        .sum();
    let uniform_expected_prior_reward =
        prior_rewards.iter().sum::<f32>() / eligible_actions.len() as f32;
    let rewarding_action_probability = eligible_actions
        .iter()
        .zip(&actual_rewards)
        .filter_map(|(action, rewarded)| (*rewarded != 0).then_some(policy_probabilities[*action]))
        .sum();
    let policy_reward_alignment = PolicyRewardAlignment {
        policy_expected_prior_reward,
        uniform_expected_prior_reward,
        policy_argmax_prior_reward: prior_rewards[policy_argmax_index],
        best_prior_reward: prior_rewards[prior_argmax_index],
        argmax_agreement: usize::from(policy_argmax_index == prior_argmax_index),
        has_rewarding_action: usize::from(rewarding_action_count != 0),
        rewarding_action_probability,
        uniform_rewarding_action_probability: rewarding_action_count as f32
            / eligible_actions.len() as f32,
        policy_argmax_rewarded: actual_rewards[policy_argmax_index],
        prior_argmax_rewarded: actual_rewards[prior_argmax_index],
    };
    samples.push(Sample {
        representations,
        position: y * WIDTH + x,
        dense_right: usize::from(x >= WIDTH - 2),
        food: environment.food_index(),
        rewarded: usize::from(rewarded),
        reward_prediction: Some(reward_prediction),
        posterior_observation_mse: Some(posterior_observation_mse),
        next_rewarded: None,
        prior_reward_prediction: None,
        prior_reward_rollout: Vec::new(),
        prior_observation_rollout: Vec::new(),
        invalid_action_probability: Some(invalid_action_probability),
        unmasked_argmax_invalid: Some(usize::from(!action_mask[unmasked_argmax])),
        policy_reward_alignment: Some(policy_reward_alignment),
    });
}

fn score_pending_observations(
    pending: &mut Vec<PendingObservationRollout>,
    samples: &mut [Sample],
    actual: &[f32],
) {
    for rollout in pending.iter_mut() {
        let predicted = &rollout.predictions[rollout.next_horizon];
        samples[rollout.sample_index]
            .prior_observation_rollout
            .push(ObservationPredictionError {
                model_mse: mean_squared_error(predicted, actual),
                persistence_mse: mean_squared_error(&rollout.persistence, actual),
            });
        rollout.next_horizon += 1;
    }
    pending.retain(|rollout| rollout.next_horizon < rollout.predictions.len());
}

fn mean_squared_error(prediction: &[f32], target: &[f32]) -> f32 {
    assert_eq!(prediction.len(), target.len());
    prediction
        .iter()
        .zip(target)
        .map(|(prediction, target)| {
            let error = prediction - target;
            error * error
        })
        .sum::<f32>()
        / prediction.len() as f32
}

fn random_action(environment: &GridWorld, all_actions: bool, rng: &mut StdRng) -> usize {
    if all_actions {
        return rng.random_range(0..ACTION_COUNT);
    }
    let mask = environment.action_mask().expect("GridWorld action mask");
    let valid = mask
        .iter()
        .enumerate()
        .filter_map(|(action, enabled)| enabled.then_some(action))
        .collect::<Vec<_>>();
    valid[rng.random_range(0..valid.len())]
}

fn label_probes(samples: &[Sample], representation: usize) -> serde_json::Value {
    json!({
        "position": centroid_accuracy(
            samples,
            representation,
            WIDTH * HEIGHT,
            |sample| sample.position,
        ),
        "dense_right": centroid_accuracy(
            samples,
            representation,
            2,
            |sample| sample.dense_right,
        ),
        "food": centroid_accuracy(
            samples,
            representation,
            3,
            |sample| sample.food,
        ),
        "reward": centroid_accuracy(
            samples,
            representation,
            2,
            |sample| sample.rewarded,
        ),
    })
}

fn centroid_accuracy(
    samples: &[Sample],
    representation: usize,
    class_count: usize,
    label: impl Fn(&Sample) -> usize,
) -> serde_json::Value {
    let feature_dim = samples[0].representations[representation].len();
    assert!(feature_dim > 0);
    assert!(
        samples
            .iter()
            .all(|sample| sample.representations[representation].len() == feature_dim)
    );
    let mut sums = vec![vec![0.0_f32; feature_dim]; class_count];
    let mut counts = vec![0_usize; class_count];
    for (index, sample) in samples.iter().enumerate() {
        if index % 5 == 0 {
            continue;
        }
        let class = label(sample);
        counts[class] += 1;
        for (sum, value) in sums[class]
            .iter_mut()
            .zip(&sample.representations[representation])
        {
            *sum += value;
        }
    }
    for (sum, count) in sums.iter_mut().zip(&counts) {
        if *count > 0 {
            for value in sum {
                *value /= *count as f32;
            }
        }
    }

    let mut correct = 0;
    let mut tested = 0;
    let mut test_counts = vec![0_usize; class_count];
    let mut correct_by_class = vec![0_usize; class_count];
    for (index, sample) in samples.iter().enumerate() {
        if index % 5 != 0 {
            continue;
        }
        let predicted = sums
            .iter()
            .enumerate()
            .filter(|(class, _)| counts[*class] > 0)
            .min_by(|(_, left), (_, right)| {
                squared_distance(&sample.representations[representation], left).total_cmp(
                    &squared_distance(&sample.representations[representation], right),
                )
            })
            .map(|(class, _)| class)
            .expect("at least one trained centroid");
        let actual = label(sample);
        let matched = usize::from(predicted == actual);
        correct += matched;
        test_counts[actual] += 1;
        correct_by_class[actual] += matched;
        tested += 1;
    }

    let recalls = correct_by_class
        .iter()
        .zip(&test_counts)
        .map(|(correct, count)| (*count > 0).then(|| *correct as f32 / *count as f32))
        .collect::<Vec<_>>();
    let balanced_accuracy = recalls.iter().flatten().sum::<f32>()
        / recalls.iter().filter(|recall| recall.is_some()).count() as f32;

    json!({
        "accuracy": correct as f32 / tested as f32,
        "balanced_accuracy": balanced_accuracy,
        "correct": correct,
        "tested": tested,
        "train_class_counts": counts,
        "test_class_counts": test_counts,
        "recall_by_class": recalls,
        "observed_classes": counts.iter().filter(|count| **count > 0).count(),
        "class_count": class_count,
    })
}

fn squared_distance(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let difference = left - right;
            difference * difference
        })
        .sum()
}

fn reward_prediction_metrics(
    samples: &[Sample],
    label: impl Fn(&Sample) -> usize,
) -> serde_json::Value {
    prediction_metrics(
        samples
            .iter()
            .map(|sample| {
                (
                    sample
                        .reward_prediction
                        .expect("checkpoint samples have reward predictions"),
                    label(sample),
                )
            })
            .collect(),
    )
}

fn prior_reward_prediction_metrics(samples: &[Sample]) -> serde_json::Value {
    prediction_metrics(
        samples
            .iter()
            .filter_map(|sample| sample.prior_reward_prediction.zip(sample.next_rewarded))
            .collect(),
    )
}

fn rollout_reward_prediction_metrics(
    samples: &[Sample],
    horizon: usize,
) -> Option<serde_json::Value> {
    let scored = samples
        .iter()
        .filter_map(|sample| sample.prior_reward_rollout.get(horizon).copied())
        .collect::<Vec<_>>();
    (!scored.is_empty()).then(|| prediction_metrics(scored))
}

fn rollout_observation_prediction_metrics(
    samples: &[Sample],
    horizon: usize,
) -> Option<serde_json::Value> {
    let scored = samples
        .iter()
        .filter_map(|sample| sample.prior_observation_rollout.get(horizon).copied())
        .collect::<Vec<_>>();
    if scored.is_empty() {
        return None;
    }
    let count = scored.len();
    let model_mse = scored.iter().map(|error| error.model_mse).sum::<f32>() / count as f32;
    let persistence_mse = scored
        .iter()
        .map(|error| error.persistence_mse)
        .sum::<f32>()
        / count as f32;
    let relative_mse_reduction = (persistence_mse > 0.0).then(|| 1.0 - model_mse / persistence_mse);
    let model_better_count = scored
        .iter()
        .filter(|error| error.model_mse < error.persistence_mse)
        .count();
    Some(json!({
        "mean_squared_error": model_mse,
        "root_mean_squared_error": model_mse.sqrt(),
        "persistence_mean_squared_error": persistence_mse,
        "persistence_root_mean_squared_error": persistence_mse.sqrt(),
        "relative_mse_reduction": relative_mse_reduction,
        "model_better_rate": model_better_count as f32 / count as f32,
        "sample_count": count,
    }))
}

fn posterior_observation_prediction_metrics(samples: &[Sample]) -> Option<serde_json::Value> {
    let errors = samples
        .iter()
        .filter_map(|sample| sample.posterior_observation_mse)
        .collect::<Vec<_>>();
    if errors.is_empty() {
        return None;
    }
    let mean_squared_error = errors.iter().sum::<f32>() / errors.len() as f32;
    Some(json!({
        "mean_squared_error": mean_squared_error,
        "root_mean_squared_error": mean_squared_error.sqrt(),
        "sample_count": errors.len(),
    }))
}

fn policy_action_mask_metrics(samples: &[Sample]) -> Option<serde_json::Value> {
    let scored = samples
        .iter()
        .filter_map(|sample| {
            sample
                .invalid_action_probability
                .zip(sample.unmasked_argmax_invalid)
        })
        .collect::<Vec<_>>();
    if scored.is_empty() {
        return None;
    }
    let sample_count = scored.len();
    let invalid_probability_mean = scored
        .iter()
        .map(|(probability, _)| probability)
        .sum::<f32>()
        / sample_count as f32;
    let invalid_argmax_count = scored.iter().map(|(_, invalid)| invalid).sum::<usize>();
    Some(json!({
        "invalid_probability_mean": invalid_probability_mean,
        "invalid_argmax_rate": invalid_argmax_count as f32 / sample_count as f32,
        "invalid_argmax_count": invalid_argmax_count,
        "sample_count": sample_count,
    }))
}

fn policy_reward_alignment_metrics(samples: &[Sample]) -> Option<serde_json::Value> {
    let scored = samples
        .iter()
        .filter_map(|sample| sample.policy_reward_alignment)
        .collect::<Vec<_>>();
    if scored.is_empty() {
        return None;
    }
    let mean = |value: fn(&PolicyRewardAlignment) -> f32| {
        scored.iter().map(value).sum::<f32>() / scored.len() as f32
    };
    let rewarding = scored
        .iter()
        .filter(|sample| sample.has_rewarding_action != 0)
        .collect::<Vec<_>>();
    let rewarding_metrics = (!rewarding.is_empty()).then(|| {
        let mean = |value: fn(&PolicyRewardAlignment) -> f32| {
            rewarding.iter().map(|sample| value(sample)).sum::<f32>() / rewarding.len() as f32
        };
        json!({
            "policy_rewarding_action_probability_mean": mean(
                |sample| sample.rewarding_action_probability,
            ),
            "uniform_rewarding_action_probability_mean": mean(
                |sample| sample.uniform_rewarding_action_probability,
            ),
            "policy_argmax_reward_rate": mean(|sample| sample.policy_argmax_rewarded as f32),
            "prior_argmax_reward_rate": mean(|sample| sample.prior_argmax_rewarded as f32),
        })
    });
    Some(json!({
        "sample_count": scored.len(),
        "policy_expected_prior_reward_mean": mean(
            |sample| sample.policy_expected_prior_reward,
        ),
        "uniform_expected_prior_reward_mean": mean(
            |sample| sample.uniform_expected_prior_reward,
        ),
        "policy_argmax_prior_reward_mean": mean(
            |sample| sample.policy_argmax_prior_reward,
        ),
        "best_prior_reward_mean": mean(|sample| sample.best_prior_reward),
        "policy_prior_argmax_agreement_rate": mean(|sample| sample.argmax_agreement as f32),
        "rewarding_state_count": rewarding.len(),
        "rewarding_states": rewarding_metrics,
    }))
}

fn prediction_metrics(mut scored: Vec<(f32, usize)>) -> serde_json::Value {
    assert!(!scored.is_empty());
    let mut prediction_sums = [0.0_f32; 2];
    let mut counts = [0_usize; 2];
    let mut correct = [0_usize; 2];
    let mut absolute_error = 0.0_f32;
    for &(prediction, actual) in &scored {
        assert!(actual < 2);
        assert!(prediction.is_finite());
        prediction_sums[actual] += prediction;
        counts[actual] += 1;
        correct[actual] += usize::from(usize::from(prediction >= 0.5) == actual);
        absolute_error += (prediction - actual as f32).abs();
    }
    let prediction_means = [
        (counts[0] > 0).then(|| prediction_sums[0] / counts[0] as f32),
        (counts[1] > 0).then(|| prediction_sums[1] / counts[1] as f32),
    ];
    let recalls = [
        (counts[0] > 0).then(|| correct[0] as f32 / counts[0] as f32),
        (counts[1] > 0).then(|| correct[1] as f32 / counts[1] as f32),
    ];
    let balanced_accuracy = recalls.iter().flatten().sum::<f32>()
        / recalls.iter().filter(|recall| recall.is_some()).count() as f32;
    let prediction_gap = prediction_means[1]
        .zip(prediction_means[0])
        .map(|(positive, negative)| positive - negative);
    let (roc_auc, average_precision, best_balanced_accuracy, best_threshold) =
        ranking_metrics(&mut scored, counts);
    let sample_count = scored.len();
    json!({
        "accuracy": (correct[0] + correct[1]) as f32 / sample_count as f32,
        "balanced_accuracy": balanced_accuracy,
        "best_balanced_accuracy": best_balanced_accuracy,
        "best_threshold": best_threshold,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "mean_absolute_error": absolute_error / sample_count as f32,
        "negative_prediction_mean": prediction_means[0],
        "positive_prediction_mean": prediction_means[1],
        "prediction_gap": prediction_gap,
        "class_counts": counts,
        "recall_by_class": recalls,
    })
}

fn ranking_metrics(
    scored: &mut [(f32, usize)],
    counts: [usize; 2],
) -> (Option<f32>, Option<f32>, Option<f32>, Option<f32>) {
    if counts.contains(&0) {
        return (None, None, None, None);
    }

    let mut concordance = 0.0_f32;
    for &(positive, label) in scored.iter() {
        if label != 1 {
            continue;
        }
        for &(negative, other_label) in scored.iter() {
            if other_label != 0 {
                continue;
            }
            concordance += match positive.total_cmp(&negative) {
                std::cmp::Ordering::Greater => 1.0,
                std::cmp::Ordering::Equal => 0.5,
                std::cmp::Ordering::Less => 0.0,
            };
        }
    }
    let roc_auc = concordance / (counts[0] * counts[1]) as f32;

    scored.sort_by(|left, right| right.0.total_cmp(&left.0));
    let mut seen = 0_usize;
    let mut positive_seen = 0_usize;
    let mut average_precision_sum = 0.0_f32;
    let mut best_balanced_accuracy = 0.0_f32;
    let mut best_threshold = scored[0].0;
    let mut start = 0;
    while start < scored.len() {
        let threshold = scored[start].0;
        let mut end = start + 1;
        while end < scored.len() && scored[end].0.total_cmp(&threshold) == std::cmp::Ordering::Equal
        {
            end += 1;
        }
        let group_positives = scored[start..end]
            .iter()
            .filter(|(_, label)| *label == 1)
            .count();
        seen += end - start;
        positive_seen += group_positives;
        let precision = positive_seen as f32 / seen as f32;
        average_precision_sum += group_positives as f32 * precision;

        let true_positive_rate = positive_seen as f32 / counts[1] as f32;
        let false_positives = seen - positive_seen;
        let true_negative_rate = (counts[0] - false_positives) as f32 / counts[0] as f32;
        let balanced = 0.5 * (true_positive_rate + true_negative_rate);
        if balanced > best_balanced_accuracy {
            best_balanced_accuracy = balanced;
            best_threshold = threshold;
        }
        start = end;
    }
    (
        Some(roc_auc),
        Some(average_precision_sum / counts[1] as f32),
        Some(best_balanced_accuracy),
        Some(best_threshold),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_eigenvalues_match_symmetric_matrix() {
        let mut eigenvalues = symmetric_eigenvalues(vec![2.0, 1.0, 1.0, 2.0], 2);
        eigenvalues.sort_by(|left, right| right.total_cmp(left));
        assert!((eigenvalues[0] - 3.0).abs() < 1e-12);
        assert!((eigenvalues[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn affine_rank_floor_recognizes_rank_one_patch_data() {
        let mut statistics = DinoPatchStatistics::new();
        let mut positive = [0.0_f32; OBSERVATION_CHANNELS];
        positive[0] = 1.0;
        let mut negative = positive;
        negative[0] = -1.0;
        statistics.add_observation(&positive);
        statistics.add_observation(&negative);
        let metrics = statistics.metrics();
        let floors = metrics["rank_floors"].as_array().unwrap();
        let rank_zero = floors[0]["minimum_mean_squared_error"].as_f64().unwrap();
        let rank_four = floors[1]["minimum_mean_squared_error"].as_f64().unwrap();
        assert!((rank_zero - 1.0 / OBSERVATION_CHANNELS as f64).abs() < 1e-12);
        assert!(rank_four < 1e-12);
    }

    fn sample(label: usize, prediction: f32) -> Sample {
        Sample {
            representations: vec![vec![0.0]],
            position: 0,
            dense_right: label,
            food: 0,
            rewarded: label,
            reward_prediction: Some(prediction),
            posterior_observation_mse: None,
            next_rewarded: None,
            prior_reward_prediction: None,
            prior_reward_rollout: Vec::new(),
            prior_observation_rollout: Vec::new(),
            invalid_action_probability: None,
            unmasked_argmax_invalid: None,
            policy_reward_alignment: None,
        }
    }

    #[test]
    fn reward_ranking_metrics_recognize_perfect_separation() {
        let samples = [
            sample(0, 0.01),
            sample(1, 0.20),
            sample(0, 0.02),
            sample(1, 0.10),
        ];
        let metrics = reward_prediction_metrics(&samples, |sample| sample.rewarded);
        assert_eq!(metrics["roc_auc"], 1.0);
        assert_eq!(metrics["average_precision"], 1.0);
        assert_eq!(metrics["best_balanced_accuracy"], 1.0);
    }

    #[test]
    fn tied_reward_predictions_have_chance_auc() {
        let samples = [sample(0, 0.02), sample(1, 0.02)];
        let metrics = reward_prediction_metrics(&samples, |sample| sample.rewarded);
        assert_eq!(metrics["roc_auc"], 0.5);
        assert_eq!(metrics["average_precision"], 0.5);
    }

    #[test]
    fn prior_reward_metrics_align_executed_transitions() {
        let mut negative = sample(0, 0.0);
        negative.prior_reward_prediction = Some(0.1);
        negative.next_rewarded = Some(0);
        let mut positive = sample(0, 0.0);
        positive.prior_reward_prediction = Some(0.9);
        positive.next_rewarded = Some(1);
        let trailing = sample(0, 0.0);
        let metrics = prior_reward_prediction_metrics(&[negative, positive, trailing]);
        assert_eq!(metrics["class_counts"], json!([1, 1]));
        assert_eq!(metrics["roc_auc"], 1.0);
    }

    #[test]
    fn observation_rollout_metrics_compare_against_persistence() {
        let mut first = sample(0, 0.0);
        first.posterior_observation_mse = Some(1.0);
        first
            .prior_observation_rollout
            .push(ObservationPredictionError {
                model_mse: 1.0,
                persistence_mse: 2.0,
            });
        let mut second = sample(0, 0.0);
        second.posterior_observation_mse = Some(3.0);
        second
            .prior_observation_rollout
            .push(ObservationPredictionError {
                model_mse: 3.0,
                persistence_mse: 6.0,
            });
        let samples = [first, second];
        let metrics = rollout_observation_prediction_metrics(&samples, 0).unwrap();
        assert_eq!(metrics["mean_squared_error"], 2.0);
        assert_eq!(metrics["persistence_mean_squared_error"], 4.0);
        assert_eq!(metrics["relative_mse_reduction"], 0.5);
        assert_eq!(metrics["model_better_rate"], 1.0);
        let posterior = posterior_observation_prediction_metrics(&samples).unwrap();
        assert_eq!(posterior["mean_squared_error"], 2.0);
    }

    #[test]
    fn policy_mask_metrics_report_invalid_mass_and_argmaxes() {
        let mut invalid = sample(0, 0.0);
        invalid.invalid_action_probability = Some(0.25);
        invalid.unmasked_argmax_invalid = Some(1);
        let mut valid = sample(0, 0.0);
        valid.invalid_action_probability = Some(0.05);
        valid.unmasked_argmax_invalid = Some(0);
        let metrics = policy_action_mask_metrics(&[invalid, valid]).unwrap();
        assert!((metrics["invalid_probability_mean"].as_f64().unwrap() - 0.15).abs() < 1e-6);
        assert_eq!(metrics["invalid_argmax_rate"], 0.5);
        assert_eq!(metrics["invalid_argmax_count"], 1);
    }

    #[test]
    fn policy_reward_alignment_separates_model_and_actor_choices() {
        let mut rewarding = sample(0, 0.0);
        rewarding.policy_reward_alignment = Some(PolicyRewardAlignment {
            policy_expected_prior_reward: 0.2,
            uniform_expected_prior_reward: 0.1,
            policy_argmax_prior_reward: 0.3,
            best_prior_reward: 0.4,
            argmax_agreement: 0,
            has_rewarding_action: 1,
            rewarding_action_probability: 0.75,
            uniform_rewarding_action_probability: 0.25,
            policy_argmax_rewarded: 1,
            prior_argmax_rewarded: 0,
        });
        let mut ordinary = sample(0, 0.0);
        ordinary.policy_reward_alignment = Some(PolicyRewardAlignment {
            policy_expected_prior_reward: 0.0,
            uniform_expected_prior_reward: 0.0,
            policy_argmax_prior_reward: 0.0,
            best_prior_reward: 0.0,
            argmax_agreement: 1,
            has_rewarding_action: 0,
            rewarding_action_probability: 0.0,
            uniform_rewarding_action_probability: 0.0,
            policy_argmax_rewarded: 0,
            prior_argmax_rewarded: 0,
        });
        let metrics = policy_reward_alignment_metrics(&[rewarding, ordinary]).unwrap();
        assert!(
            (metrics["policy_expected_prior_reward_mean"]
                .as_f64()
                .unwrap()
                - 0.1)
                .abs()
                < 1e-6,
        );
        assert_eq!(metrics["policy_prior_argmax_agreement_rate"], 0.5);
        assert_eq!(metrics["rewarding_state_count"], 1);
        assert_eq!(
            metrics["rewarding_states"]["policy_rewarding_action_probability_mean"],
            0.75,
        );
        assert_eq!(metrics["rewarding_states"]["prior_argmax_reward_rate"], 0.0,);
    }

    #[test]
    fn all_action_sampling_includes_border_noops() {
        let mut environment = GridWorld::new();
        environment.reset();
        let mut rng = StdRng::seed_from_u64(7);
        let mut seen = [false; ACTION_COUNT];
        for _ in 0..100 {
            seen[random_action(&environment, true, &mut rng)] = true;
        }
        assert!(seen.into_iter().all(|sampled| sampled));
    }
}
