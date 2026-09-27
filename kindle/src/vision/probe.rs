//! Small GPU regressors for offline representation evaluation, not agent inputs.

use meganeura::{Graph, Mode, Session, SessionConfig, nn};
use rand::{Rng, SeedableRng, rngs::StdRng};

pub struct RegressionProbe {
    session: Session,
    batch: usize,
    inputs: usize,
    hidden: usize,
    targets: usize,
    parameters: Vec<String>,
}

fn validate_dimensions(
    batch: usize,
    inputs: usize,
    hidden: usize,
    targets: usize,
) -> Result<(), &'static str> {
    if batch == 0
        || batch > 1024
        || inputs == 0
        || inputs > 65536
        || hidden == 0
        || hidden > 256
        || targets == 0
        || targets > 64
    {
        return Err("invalid offline probe dimensions");
    }
    Ok(())
}

fn graph(batch: usize, inputs: usize, hidden: usize, targets: usize) -> Graph {
    let mut g = Graph::new();
    let x = g.input("x", &[batch, inputs]);
    let y = g.input("y", &[batch, targets]);
    let mask = g.input("mask", &[batch, targets]);
    let regularization = g.input("regularization", &[1]);
    let first = nn::Linear::new(&mut g, "hidden", inputs, hidden);
    let second = nn::Linear::new(&mut g, "output", hidden, targets);
    let x = first.forward(&mut g, x);
    let x = g.relu(x);
    let prediction = second.forward(&mut g, x);
    let negative_y = g.neg(y);
    let error = g.add(prediction, negative_y);
    let square = g.mul(error, error);
    let weighted = g.mul(square, mask);
    let error = g.mean_all(weighted);
    let first_square = g.mul(first.weight, first.weight);
    let second_square = g.mul(second.weight, second.weight);
    let first_norm = g.sum_all(first_square);
    let second_norm = g.sum_all(second_square);
    let norm = g.add(first_norm, second_norm);
    let penalty = g.mul(norm, regularization);
    let loss = g.add(error, penalty);
    g.set_outputs(vec![loss, prediction]);
    g
}

impl RegressionProbe {
    pub fn new(
        batch: usize,
        inputs: usize,
        hidden: usize,
        targets: usize,
        seed: u64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        validate_dimensions(batch, inputs, hidden, targets)?;
        let gpu = std::sync::Arc::new(crate::init_gpu_context()?);
        Ok(Self::build(batch, inputs, hidden, targets, seed, gpu))
    }

    fn build(
        batch: usize,
        inputs: usize,
        hidden: usize,
        targets: usize,
        seed: u64,
        gpu: std::sync::Arc<blade_graphics::Context>,
    ) -> Self {
        let graph = graph(batch, inputs, hidden, targets);
        let mut session = meganeura::build(
            &graph,
            SessionConfig {
                mode: Mode::Training,
                gpu: Some(gpu),
                ..SessionConfig::default()
            },
        )
        .0;
        let mut rng = StdRng::seed_from_u64(seed);
        for (prefix, input, output) in [("hidden", inputs, hidden), ("output", hidden, targets)] {
            let bound = (6.0 / (input + output) as f32).sqrt();
            let values = (0..input * output)
                .map(|_| rng.random_range(-bound..bound))
                .collect::<Vec<_>>();
            session.set_parameter(&format!("{prefix}.weight"), &values);
            session.set_parameter(&format!("{prefix}.bias"), &vec![0.0; output]);
        }
        let mut parameters = session
            .param_names()
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        parameters.sort();
        Self {
            session,
            batch,
            inputs,
            hidden,
            targets,
            parameters,
        }
    }

    /// Independent weights and optimizer state on the same caller-owned device.
    /// Probe sweeps need many fresh models, not many Vulkan device lifetimes.
    pub fn reset(&mut self, inputs: usize, targets: usize, seed: u64) -> Result<(), &'static str> {
        validate_dimensions(self.batch, inputs, self.hidden, targets)?;
        *self = Self::build(
            self.batch,
            inputs,
            self.hidden,
            targets,
            seed,
            self.session.context(),
        );
        Ok(())
    }

    pub fn gpu_device(&self) -> crate::GpuDeviceInfo {
        crate::gpu_device_info(self.session.device_information())
    }

    pub fn gpu_memory_budget(&self) -> Option<crate::GpuMemoryBudget> {
        self.session
            .device_memory_stats()
            .map(|s| crate::GpuMemoryBudget {
                usage_bytes: s.usage_bytes,
                budget_bytes: s.budget_bytes,
            })
    }

    pub fn predict(&mut self, features: &[f32]) -> Result<Vec<f32>, &'static str> {
        self.inputs(
            features,
            &vec![0.0; self.batch * self.targets],
            &vec![0.0; self.batch * self.targets],
            0.0,
        )?;
        // No persistent optimizer is configured. Evaluation may compute unused
        // gradients but cannot update weights, moments or optimizer age.
        self.session.step();
        self.session.wait();
        let mut predictions = vec![0.0; self.batch * self.targets];
        self.session.read_output_by_index(1, &mut predictions);
        Ok(predictions)
    }

    pub fn learn(
        &mut self,
        features: &[f32],
        targets: &[f32],
        mask: &[f32],
        learning_rate: f32,
        regularization: f32,
    ) -> Result<f32, &'static str> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err("learning rate must be finite and positive");
        }
        self.inputs(features, targets, mask, regularization)?;
        self.session.step();
        self.session.wait();
        let loss = self.session.read_loss();
        if !loss.is_finite() {
            return Err("nonfinite probe loss");
        }
        self.session.adam_step(learning_rate, 0.9, 0.999, 1e-8);
        self.session.wait();
        Ok(loss)
    }

    fn inputs(
        &mut self,
        features: &[f32],
        targets: &[f32],
        mask: &[f32],
        regularization: f32,
    ) -> Result<(), &'static str> {
        if features.len() != self.batch * self.inputs
            || targets.len() != self.batch * self.targets
            || mask.len() != targets.len()
            || !regularization.is_finite()
            || regularization < 0.0
            || !features.iter().chain(targets).all(|v| v.is_finite())
            || !mask.iter().all(|v| v.is_finite() && *v >= 0.0)
        {
            return Err("wrong or nonfinite probe batch");
        }
        self.session.set_input("x", features);
        self.session.set_input("y", targets);
        self.session.set_input("mask", mask);
        self.session.set_input("regularization", &[regularization]);
        Ok(())
    }

    pub fn parameters(&self) -> Vec<Vec<f32>> {
        self.session.read_params(
            &self
                .parameters
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
        )
    }

    /// Restore selected weights for final evaluation, not for training resume.
    pub fn set_parameters(&mut self, values: &[Vec<f32>]) -> Result<(), &'static str> {
        if values.len() != self.parameters.len()
            || values.iter().zip(&self.parameters).any(|(v, n)| {
                self.session.param_size(n) != Some(v.len()) || !v.iter().all(|x| x.is_finite())
            })
        {
            return Err("wrong probe parameters");
        }
        for (name, values) in self.parameters.iter().zip(values) {
            self.session.set_parameter(name, values);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_dimensions_precede_gpu_initialization() {
        for (batch, inputs, hidden, targets) in
            [(0, 2, 2, 2), (2, 0, 2, 2), (2, 2, 0, 2), (2, 2, 2, 65)]
        {
            assert!(RegressionProbe::new(batch, inputs, hidden, targets, 0).is_err());
        }
    }

    #[test]
    #[ignore = "requires GPU; tiny independent offline regression check"]
    fn tiny_probe_matches_scalar_formula_and_learns_without_eval_updates() {
        let mut probe = RegressionProbe::new(4, 2, 2, 1, 43).unwrap();
        let initial = probe.parameters();
        let device = probe.session.context();
        let hardware_check = |probe: &RegressionProbe| {
            if let Ok(expected) = std::env::var("KINDLE_EXPECT_DEVICE_NAME") {
                assert_eq!(probe.gpu_device().device_name, expected);
                assert!(!probe.gpu_device().is_software_emulated);
                let memory = probe.gpu_memory_budget().expect("Vulkan memory budget");
                assert!(memory.budget_bytes - memory.usage_bytes >= 2 << 30);
                eprintln!(
                    "offline probe device={:?} memory={memory:?}",
                    probe.gpu_device()
                );
            }
        };
        hardware_check(&probe);
        // Sorted: hidden.bias, hidden.weight, output.bias, output.weight.
        probe
            .set_parameters(&[
                vec![0.5, -0.25],
                vec![1.0, -1.0, 0.5, 2.0],
                vec![0.1],
                vec![0.75, -0.5],
            ])
            .unwrap();
        let x = [0.5, 1.0, -0.5, 0.2, 1.0, -1.0, 0.0, 0.0];
        let expected = x
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&[a, b]| {
                0.75 * (a + 0.5 * b + 0.5_f32).max(0.0) - 0.5 * (-a + 2.0 * b - 0.25_f32).max(0.0)
                    + 0.1
            })
            .collect::<Vec<_>>();
        let before = probe.parameters();
        for (a, b) in probe.predict(&x).unwrap().iter().zip(&expected) {
            assert!((a - b).abs() < 1e-5);
        }
        assert_eq!(before, probe.parameters());
        let targets = [0.2, 0.4, -0.6, 0.1];
        let mask = [1.0, 0.0, 0.5, 2.0];
        let regularization = 0.001_f32;
        probe.inputs(&x, &targets, &mask, regularization).unwrap();
        probe.session.step();
        probe.session.wait();
        let mut reference = [vec![0.0_f64; 2], vec![0.0; 4], vec![0.0; 1], vec![0.0; 2]];
        for row in 0..4 {
            let [a, b] = [f64::from(x[2 * row]), f64::from(x[2 * row + 1])];
            let hidden = [a + 0.5 * b + 0.5, -a + 2.0 * b - 0.25];
            let delta = 2.0 * f64::from(expected[row] - targets[row]) * f64::from(mask[row]) / 4.0;
            reference[2][0] += delta;
            for lane in 0..2 {
                reference[3][lane] += delta * hidden[lane].max(0.0);
                if hidden[lane] > 0.0 {
                    let gradient = delta * f64::from(before[3][lane]);
                    reference[0][lane] += gradient;
                    reference[1][lane] += a * gradient;
                    reference[1][2 + lane] += b * gradient;
                }
            }
        }
        for index in [1, 3] {
            for (value, weight) in reference[index].iter_mut().zip(&before[index]) {
                *value += 2.0 * f64::from(regularization) * f64::from(*weight);
            }
        }
        for (name, expected) in probe.parameters.iter().zip(&reference) {
            let mut actual = vec![0.0; expected.len()];
            probe.session.read_param_grad(name, &mut actual);
            for (a, b) in actual.iter().zip(expected) {
                assert!(
                    (f64::from(*a) - b).abs() < 2e-5 * (1.0 + b.abs()),
                    "{name}: {a} != {b}"
                );
            }
        }
        let first = probe.learn(&x, &targets, &[1.0; 4], 0.01, 0.0).unwrap();
        let mut last = first;
        for _ in 0..200 {
            last = probe.learn(&x, &targets, &[1.0; 4], 0.01, 0.0).unwrap();
        }
        assert!(last < 0.1 * first, "{first} -> {last}");
        let trained = probe.parameters();
        let names = probe.parameters.clone();
        let names = names.iter().map(String::as_str).collect::<Vec<_>>();
        let moments = probe.session.read_adam_states(&names);
        let age = probe.session.adam_step_count();
        probe.predict(&x).unwrap();
        assert_eq!(trained, probe.parameters());
        assert_eq!(moments, probe.session.read_adam_states(&names));
        assert_eq!(age, probe.session.adam_step_count());
        probe.reset(2, 1, 43).unwrap();
        assert_eq!(initial, probe.parameters());
        assert_eq!(probe.session.adam_step_count(), 0);
        probe.learn(&x, &targets, &[1.0; 4], 0.01, 0.0).unwrap();
        let fresh_step = probe.parameters();
        let fresh_moments = probe.session.read_adam_states(&names);
        // More complete model lifetimes than the failed nine-head sweep, but
        // a single physical device. Each reset is a genuinely fresh learner.
        for _ in 0..16 {
            probe.reset(3, 2, 7).unwrap();
            assert_eq!(probe.predict(&[0.0; 12]).unwrap().len(), 8);
            probe.reset(2, 1, 43).unwrap();
            assert!(std::sync::Arc::ptr_eq(&device, &probe.session.context()));
            assert_eq!(initial, probe.parameters());
            assert_eq!(probe.session.adam_step_count(), 0);
            probe.learn(&x, &targets, &[1.0; 4], 0.01, 0.0).unwrap();
            assert_eq!(fresh_step, probe.parameters());
            assert_eq!(fresh_moments, probe.session.read_adam_states(&names));
        }
        hardware_check(&probe);
    }
}
