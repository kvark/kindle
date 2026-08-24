use std::path::{Path, PathBuf};

use clap::Parser;
use kindle::vision::DinoPerception;
use kindle::{ActionMode, DreamerAgent, Environment};
use kindle_gym::grid_world::{ACTION_COUNT, GridWorld, HEIGHT, WIDTH};
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

    #[arg(long, default_value_t = 0)]
    seed: u64,
}

struct Sample {
    representations: Vec<Vec<f32>>,
    position: usize,
    food: usize,
    rewarded: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = Arguments::parse();
    if arguments.samples < 100 {
        return Err("--samples must be at least 100".into());
    }

    let (source, samples, representation_names) = if let Some(checkpoint) = &arguments.checkpoint {
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

    let mut representations = serde_json::Map::new();
    let mut feature_dims = serde_json::Map::new();
    for (index, name) in representation_names.iter().enumerate() {
        representations.insert((*name).into(), label_probes(&samples, index));
        feature_dims.insert(
            (*name).into(),
            json!(samples[0].representations[index].len()),
        );
    }
    println!(
        "{}",
        serde_json::to_string(&json!({
            "event": "representation_probe",
            "source": source,
            "samples": arguments.samples,
            "seed": arguments.seed,
            "feature_dims": feature_dims,
            "representations": representations,
        }))?
    );
    Ok(())
}

fn collect_perception_samples(
    arguments: &Arguments,
) -> Result<Vec<Sample>, Box<dyn std::error::Error>> {
    let mut perception = DinoPerception::load_vits16(&arguments.dino_checkpoint, None, None)?;
    let mut environment = GridWorld::new();
    let mut rng = StdRng::seed_from_u64(arguments.seed);
    environment.reset();
    let mut samples = Vec::with_capacity(arguments.samples);
    let mut rewarded = false;

    for _ in 0..arguments.samples {
        let frame = environment.render();
        let (x, y) = environment.position();
        samples.push(Sample {
            representations: vec![
                perception
                    .encode_frame_rgb8(frame.pixels(), frame.width(), frame.height())
                    .as_slice()
                    .to_vec(),
            ],
            position: y * WIDTH + x,
            food: environment.food_index(),
            rewarded: usize::from(rewarded),
        });

        let action = random_valid_action(&environment, &mut rng);
        let transition = environment.step(action);
        rewarded = transition.reward.extrinsic != 0.0;
        if transition.terminated || transition.truncated {
            environment.reset();
            rewarded = false;
        }
    }
    Ok(samples)
}

fn collect_latent_samples(
    arguments: &Arguments,
    checkpoint: &Path,
) -> Result<Vec<Sample>, Box<dyn std::error::Error>> {
    let mut agent = DreamerAgent::restore(checkpoint, &arguments.dino_checkpoint, None)?;
    let mut environment = GridWorld::new();
    let mut rng = StdRng::seed_from_u64(arguments.seed);
    let first = environment.reset();
    agent.begin_episode(&first);
    let mut samples = Vec::with_capacity(arguments.samples);
    push_latent_sample(&mut samples, &agent, &environment, false);

    while samples.len() < arguments.samples {
        let action = random_valid_action(&environment, &mut rng);
        let mut forced_mask = [false; ACTION_COUNT];
        forced_mask[action] = true;
        assert_eq!(agent.act(ActionMode::Sample, Some(&forced_mask)), action);
        let transition = environment.step(action);
        let rewarded = transition.reward.extrinsic != 0.0;
        let ended = transition.terminated || transition.truncated;
        agent.observe(&transition);
        push_latent_sample(&mut samples, &agent, &environment, rewarded);
        if ended && samples.len() < arguments.samples {
            agent.begin_episode(&environment.reset());
        }
    }
    Ok(samples)
}

fn push_latent_sample(
    samples: &mut Vec<Sample>,
    agent: &DreamerAgent,
    environment: &GridWorld,
    rewarded: bool,
) {
    let (x, y) = environment.position();
    let deter_dim = agent.core().config().network().deter;
    let latent = agent.latent_feature();
    samples.push(Sample {
        representations: vec![
            agent.encoded_observation().to_vec(),
            latent.to_vec(),
            latent[..deter_dim].to_vec(),
            latent[deter_dim..].to_vec(),
        ],
        position: y * WIDTH + x,
        food: environment.food_index(),
        rewarded: usize::from(rewarded),
    });
}

fn random_valid_action(environment: &GridWorld, rng: &mut StdRng) -> usize {
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
