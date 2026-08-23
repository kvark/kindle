use std::path::PathBuf;

use clap::Parser;
use kindle::Environment;
use kindle::vision::{DinoObservation, DinoPerception};
use kindle_gym::grid_world::{GridWorld, HEIGHT, WIDTH};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde_json::json;

#[derive(Debug, Parser)]
#[command(about = "Probe GridWorld state separability in frozen DINO features")]
struct Arguments {
    /// DINOv3 ViT-S/16 model.safetensors.
    dino_checkpoint: PathBuf,

    /// Number of frames from a random valid-action trajectory.
    #[arg(long, default_value_t = 500)]
    samples: usize,

    #[arg(long, default_value_t = 0)]
    seed: u64,
}

struct Sample {
    observation: DinoObservation,
    position: usize,
    food: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = Arguments::parse();
    if arguments.samples < 100 {
        return Err("--samples must be at least 100".into());
    }

    let mut perception = DinoPerception::load_vits16(&arguments.dino_checkpoint, None, None)?;
    let mut environment = GridWorld::new();
    let mut rng = StdRng::seed_from_u64(arguments.seed);
    environment.reset();
    let mut samples = Vec::with_capacity(arguments.samples);

    for _ in 0..arguments.samples {
        let frame = environment.render();
        let (x, y) = environment.position();
        samples.push(Sample {
            observation: perception.encode_frame_rgb8(
                frame.pixels(),
                frame.width(),
                frame.height(),
            ),
            position: y * WIDTH + x,
            food: environment.food_index(),
        });

        let mask = environment.action_mask().expect("GridWorld action mask");
        let valid = mask
            .iter()
            .enumerate()
            .filter_map(|(action, enabled)| enabled.then_some(action))
            .collect::<Vec<_>>();
        let transition = environment.step(valid[rng.random_range(0..valid.len())]);
        if transition.terminated || transition.truncated {
            environment.reset();
        }
    }

    let position = centroid_accuracy(&samples, WIDTH * HEIGHT, |sample| sample.position);
    let food = centroid_accuracy(&samples, 3, |sample| sample.food);
    println!(
        "{}",
        serde_json::to_string(&json!({
            "event": "perception_probe",
            "samples": arguments.samples,
            "seed": arguments.seed,
            "position": position,
            "food": food,
        }))?
    );
    Ok(())
}

fn centroid_accuracy(
    samples: &[Sample],
    class_count: usize,
    label: impl Fn(&Sample) -> usize,
) -> serde_json::Value {
    let mut sums = vec![vec![0.0_f32; DinoObservation::LEN]; class_count];
    let mut counts = vec![0_usize; class_count];
    for (index, sample) in samples.iter().enumerate() {
        if index % 5 == 0 {
            continue;
        }
        let class = label(sample);
        counts[class] += 1;
        for (sum, value) in sums[class].iter_mut().zip(sample.observation.as_slice()) {
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
    for (index, sample) in samples.iter().enumerate() {
        if index % 5 != 0 {
            continue;
        }
        let predicted = sums
            .iter()
            .enumerate()
            .filter(|(class, _)| counts[*class] > 0)
            .min_by(|(_, left), (_, right)| {
                squared_distance(sample.observation.as_slice(), left)
                    .total_cmp(&squared_distance(sample.observation.as_slice(), right))
            })
            .map(|(class, _)| class)
            .expect("at least one trained centroid");
        correct += usize::from(predicted == label(sample));
        tested += 1;
    }

    json!({
        "accuracy": correct as f32 / tested as f32,
        "correct": correct,
        "tested": tested,
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
