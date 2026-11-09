//! Conditional pixel denoising and generated-transition event readouts.
use super::{
    DreamerConfig, TrainingPrecision, WorldPrediction, checkpoint, distributions::TwoHotBins,
    gaussian, runtime,
};
use crate::{env::RgbFrame, vision::preprocess_gpu::GpuPreprocessor};
use meganeura::{Graph, Mode, Session, graph::NodeId};
use rand::{SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use std::{path::Path, sync::Arc};
mod network;
#[cfg(test)]
mod tests;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DenoisingConfig {
    pub action_count: usize,
    pub batch_size: usize,
    pub train_batch_size: usize,
    pub side: usize,
    pub value_bins: usize,
    pub learning_rate: f32,
    pub seed: u64,
    pub horizon: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub context_motion_scale: f32,
    #[serde(default, skip_serializing_if = "is_false")]
    pub residual_frame: bool,
}
impl Default for DenoisingConfig {
    fn default() -> Self {
        Self {
            action_count: 18,
            batch_size: 8,
            train_batch_size: 32,
            side: 64,
            value_bins: 255,
            learning_rate: 1e-4,
            seed: 41001,
            horizon: 333,
            context_motion_scale: 0.,
            residual_frame: false,
        }
    }
}
impl DenoisingConfig {
    fn validate(&self) -> Result<(), &'static str> {
        if self.action_count == 0
            || self.action_count > 64
            || self.batch_size == 0
            || self.batch_size > 32
            || self.train_batch_size == 0
            || self.train_batch_size > 64
            || !matches!(self.side, 32 | 64)
            || self.value_bins < 3
            || self.value_bins.is_multiple_of(2)
            || !self.learning_rate.is_finite()
            || self.learning_rate <= 0.
            || self.horizon < 2
            || !self.context_motion_scale.is_finite()
            || !(0.0..=1000.0).contains(&self.context_motion_scale)
        {
            return Err("invalid denoising world geometry or learning configuration");
        }
        Ok(())
    }
    fn checkpoint_format(&self) -> &'static str {
        if self.residual_frame {
            "kindle-denoising-world-v3"
        } else if self.context_motion_scale == 0. {
            "kindle-denoising-world-v1"
        } else {
            "kindle-denoising-world-v2"
        }
    }
    fn sampler_name(&self) -> &'static str {
        if self.residual_frame {
            "edm-euler3-sigma5-last-frame-residual-quantized"
        } else {
            "edm-euler3-sigma5-quantized"
        }
    }
    fn optimizer(&self) -> DreamerConfig {
        let mut c = DreamerConfig::tiny(self.action_count);
        c.training_precision = TrainingPrecision::Auto;
        c.learning_rate = self.learning_rate;
        c.optimizer_epsilon = 1e-8;
        c.optimizer_beta2 = 0.999;
        c.agc = 0.3;
        c
    }
}

const SIGMA_DATA: f32 = 0.5;
const SIGMA_MIN: f32 = 0.002;
const SIGMA_MAX: f32 = 5.;
fn is_false(value: &bool) -> bool {
    !value
}
fn is_zero(value: &f32) -> bool {
    *value == 0.
}
fn conditioners(sigma: f32) -> (f32, f32, f32, f32) {
    let variance = sigma * sigma + SIGMA_DATA * SIGMA_DATA;
    (
        variance.sqrt().recip(),
        sigma * SIGMA_DATA / variance.sqrt(),
        SIGMA_DATA * SIGMA_DATA / variance,
        sigma.ln() / 4.,
    )
}
fn schedule() -> [f32; 4] {
    let lo = SIGMA_MIN.powf(1. / 7.);
    let hi = SIGMA_MAX.powf(1. / 7.);
    [SIGMA_MAX, (hi + 0.5 * (lo - hi)).powi(7), SIGMA_MIN, 0.]
}
fn quantize(x: f32) -> f32 {
    ((x.clamp(-1., 1.) + 1.) * 127.5).round() / 127.5 - 1.
}
fn coordinate(value: f32, base: f32, residual: bool) -> f32 {
    if residual { value - base } else { value }
}
fn absolute(value: f32, base: f32, residual: bool) -> f32 {
    quantize(if residual { value + base } else { value })
}
fn valid_images(values: &[f32]) -> bool {
    values
        .iter()
        .all(|v| v.is_finite() && (-0.5..=0.5).contains(v))
}

struct History {
    rgb: Vec<f32>,
    actions: Vec<i32>,
    batch: usize,
    width: usize,
}
impl History {
    fn empty(batch: usize, width: usize) -> Self {
        Self {
            rgb: vec![0.; 4 * batch * width],
            actions: vec![-1; 4 * batch],
            batch,
            width,
        }
    }
    fn push(&mut self, image: &[f32], actions: &[i32]) {
        let b = self.batch;
        let w = self.width;
        self.rgb.copy_within(b * w.., 0);
        self.actions.copy_within(b.., 0);
        for (lane, &action) in actions.iter().enumerate() {
            if action == -1 {
                for t in 0..3 {
                    self.rgb[(t * b + lane) * w..(t * b + lane + 1) * w].fill(0.);
                    self.actions[t * b + lane] = -1;
                }
            }
            self.rgb[(3 * b + lane) * w..(3 * b + lane + 1) * w]
                .copy_from_slice(&image[lane * w..(lane + 1) * w]);
            self.actions[3 * b + lane] = action;
        }
    }
    fn condition(&self, lane: usize, incoming: i32, action_count: usize, scaled: bool) -> Vec<f32> {
        let mut result = Vec::new();
        let b = self.batch;
        let w = self.width;
        for t in 0..4 {
            result.extend(
                self.rgb[(t * b + lane) * w..(t * b + lane + 1) * w]
                    .iter()
                    .map(|&x| {
                        if incoming < 0 {
                            0.
                        } else if scaled {
                            x / SIGMA_DATA
                        } else {
                            x
                        }
                    }),
            );
        }
        let area = w / 3;
        for t in 0..4 {
            let a = if incoming < 0 {
                -1
            } else if t == 3 {
                incoming
            } else {
                self.actions[(t + 1) * b + lane]
            };
            for k in 0..action_count {
                result.extend(std::iter::repeat_n(
                    if a == k as i32 { 1. } else { 0. },
                    area,
                ));
            }
        }
        result
    }
    fn base(&self, incoming: &[i32], enabled: bool) -> Vec<f32> {
        let mut base = vec![0.; self.batch * self.width];
        if enabled {
            for (lane, &action) in incoming.iter().enumerate() {
                if action >= 0 {
                    let start = (3 * self.batch + lane) * self.width;
                    base[lane * self.width..(lane + 1) * self.width]
                        .copy_from_slice(&self.rgb[start..start + self.width]);
                }
            }
        }
        base
    }
    /// Positive, mean-one importance derived only from the conditioning history.
    fn motion_importance(&self, lane: usize, side: usize, scale: f32) -> Vec<f32> {
        let area = side * side;
        if scale == 0. || self.actions[3 * self.batch + lane] < 0 {
            return vec![1.; 3 * area];
        }
        let previous = (2 * self.batch + lane) * self.width;
        let current = (3 * self.batch + lane) * self.width;
        let motion = (0..area)
            .map(|i| {
                (0..3)
                    .map(|c| {
                        (self.rgb[current + c * area + i] - self.rgb[previous + c * area + i]).abs()
                            * 0.5
                    })
                    .fold(0., f32::max)
            })
            .collect::<Vec<_>>();
        // Separable radius-four max filter keeps nearby future motion in scope.
        let mut horizontal = vec![0.; area];
        let mut importance = vec![1.; area];
        for y in 0..side {
            for x in 0..side {
                horizontal[y * side + x] = (x.saturating_sub(4)..=(x + 4).min(side - 1))
                    .map(|xx| motion[y * side + xx])
                    .fold(0., f32::max);
            }
        }
        for y in 0..side {
            for x in 0..side {
                let change = (y.saturating_sub(4)..=(y + 4).min(side - 1))
                    .map(|yy| horizontal[yy * side + x])
                    .fold(0., f32::max);
                importance[y * side + x] += scale * change;
            }
        }
        let mean = importance.iter().sum::<f32>() / area as f32;
        for value in &mut importance {
            *value /= mean;
        }
        importance.repeat(3)
    }
}

pub struct DenoisingWorldLearner {
    config: DenoisingConfig,
    train: Session,
    sampler: Session,
    events: Session,
    gpu: Arc<blade_graphics::Context>,
    resize: Session,
    preprocessor: GpuPreprocessor,
    rng: StdRng,
    updates: u64,
    dirty: bool,
}
impl DenoisingWorldLearner {
    pub fn new(config: DenoisingConfig) -> Result<Self, Box<dyn std::error::Error>> {
        config.validate()?;
        let gpu = Arc::new(crate::init_gpu_context()?);
        let g = network::training(&config);
        let mut train = runtime::build_training_session(&g, &gpu, &config.optimizer());
        runtime::initialize_d3(&mut train, &g, config.seed);
        let mut sampler = runtime::build_learner_inference_session(
            &network::sampling(&config),
            &gpu,
            &config.optimizer(),
        );
        runtime::share_matching(&mut train, &mut sampler, "pixel.");
        let mut events = runtime::build_learner_inference_session(
            &network::event_sampling(&config),
            &gpu,
            &config.optimizer(),
        );
        runtime::share_matching(&mut train, &mut events, "pixel.");
        let mut resize_graph = Graph::new();
        let pixels = resize_graph.input(
            "observation",
            &[config.batch_size, 3 * config.side * config.side],
        );
        resize_graph.set_outputs(vec![pixels]);
        let resize = runtime::build_session(&resize_graph, &gpu, Mode::Inference, false);
        let preprocessor =
            GpuPreprocessor::rgb(Arc::clone(&gpu), config.batch_size, config.side, None);
        Ok(Self {
            resize,
            preprocessor,
            rng: StdRng::seed_from_u64(config.seed),
            config,
            train,
            sampler,
            events,
            gpu,
            updates: 0,
            dirty: false,
        })
    }
    pub fn config(&self) -> &DenoisingConfig {
        &self.config
    }
    pub fn updates(&self) -> u64 {
        self.updates
    }
    pub fn gpu_device(&self) -> crate::GpuDeviceInfo {
        crate::gpu_device_info(self.gpu.device_information())
    }
    pub fn gpu_memory_budget(&self) -> Option<crate::GpuMemoryBudget> {
        self.train
            .device_memory_stats()
            .map(|s| crate::GpuMemoryBudget {
                usage_bytes: s.usage_bytes,
                budget_bytes: s.budget_bytes,
            })
    }
    pub fn preprocess(&mut self, frames: &[RgbFrame]) -> Result<Vec<f32>, &'static str> {
        if frames.len() != self.config.batch_size {
            return Err("complete frame batch required");
        }
        let frames = frames
            .iter()
            .enumerate()
            .map(|(i, f)| (i, f.pixels(), f.width(), f.height()))
            .collect::<Vec<_>>();
        self.preprocessor.cpu_frames(&mut self.resize, &frames);
        self.resize.step();
        self.resize.wait();
        let mut pixels = vec![0.; self.config.batch_size * 3 * self.config.side * self.config.side];
        self.resize.read_output_by_index(0, &mut pixels);
        Ok(pixels)
    }
    pub fn parameter_names(&self) -> Vec<String> {
        self.train
            .param_names()
            .into_iter()
            .map(str::to_owned)
            .collect()
    }
    fn sync(&mut self) {
        if self.dirty {
            runtime::sync_matching(&self.train, &mut self.sampler, "pixel.");
            runtime::sync_matching(&self.train, &mut self.events, "pixel.");
            self.dirty = false;
        }
    }
    pub fn learn(
        &mut self,
        images: &[f32],
        actions: &[i32],
        rewards: &[f32],
        terminals: &[bool],
    ) -> Result<Vec<f32>, &'static str> {
        let c = self.config.clone();
        let b = c.train_batch_size;
        let w = 3 * c.side * c.side;
        if actions.len() < 5 * b
            || actions.len() > 64 * b
            || !actions.len().is_multiple_of(b)
            || images.len() != actions.len() * w
            || rewards.len() != actions.len()
            || terminals.len() != actions.len()
            || !valid_images(images)
            || actions
                .iter()
                .any(|&a| a < -1 || a >= c.action_count as i32)
            || rewards.iter().any(|x| !x.is_finite())
        {
            return Err("invalid denoising training sequence");
        }
        for i in b..actions.len() {
            if actions[i] == -1 && (rewards[i] != 0. || terminals[i]) {
                return Err("reset arrivals must have zero reward and no terminal target");
            }
        }
        let length = actions.len() / b;
        let mut history = History::empty(b, w);
        for t in 0..4 {
            let frame = images[t * b * w..(t + 1) * b * w]
                .iter()
                .map(|x| x * 2.)
                .collect::<Vec<_>>();
            history.push(&frame, &actions[t * b..(t + 1) * b]);
        }
        let mut report = vec![0.; 6];
        let bins = TwoHotBins::new(c.value_bins);
        for t in 4..length {
            let incoming = &actions[t * b..(t + 1) * b];
            let valid = incoming.iter().filter(|&&a| a >= 0).count();
            let actual = images[t * b * w..(t + 1) * b * w]
                .iter()
                .map(|x| x * 2.)
                .collect::<Vec<_>>();
            if valid == 0 {
                history.push(&actual, incoming);
                continue;
            }
            let sigma = gaussian::normals(&mut self.rng, b)
                .into_iter()
                .map(|n| (-1.2 + 1.2 * n).exp().clamp(SIGMA_MIN, SIGMA_MAX))
                .collect::<Vec<_>>();
            let noise = gaussian::normals(&mut self.rng, b * w);
            let base = history.base(incoming, c.residual_frame);
            let mut inputs = Vec::new();
            let mut targets = Vec::new();
            let mut event_inputs = Vec::new();
            let mut reward_targets = vec![0.; b * c.value_bins];
            let mut alive = Vec::new();
            let mut weights = Vec::new();
            let mut noisy = Vec::new();
            for lane in 0..b {
                let (cin, cout, cskip, cnoise) = conditioners(sigma[lane]);
                let x = actual[lane * w..(lane + 1) * w]
                    .iter()
                    .zip(&noise[lane * w..(lane + 1) * w])
                    .enumerate()
                    .map(|(i, (&y, &n))| {
                        let y = coordinate(y, base[lane * w + i], c.residual_frame);
                        y + sigma[lane] * n
                    })
                    .collect::<Vec<_>>();
                let cond = history.condition(lane, incoming[lane], c.action_count, true);
                inputs.extend_from_slice(&cond);
                inputs.extend(x.iter().map(|x| cin * x));
                inputs.extend(std::iter::repeat_n(cnoise, c.side * c.side));
                targets.extend(
                    actual[lane * w..(lane + 1) * w]
                        .iter()
                        .zip(&x)
                        .enumerate()
                        .map(|(i, (&y, &x))| {
                            let y = coordinate(y, base[lane * w + i], c.residual_frame);
                            (y - cskip * x) / cout
                        }),
                );
                noisy.extend(x);
                event_inputs.extend(history.condition(lane, incoming[lane], c.action_count, false));
                event_inputs.extend_from_slice(&actual[lane * w..(lane + 1) * w]);
                bins.encode(
                    rewards[t * b + lane],
                    &mut reward_targets[lane * c.value_bins..(lane + 1) * c.value_bins],
                );
                alive.push(if terminals[t * b + lane] { -0.5 } else { 0.5 });
                weights.push(if incoming[lane] >= 0 {
                    b as f32 / valid as f32
                } else {
                    0.
                });
            }
            for (name, values) in [
                ("denoiser_input", &inputs),
                ("denoiser_target", &targets),
                ("event_input", &event_inputs),
                ("reward_target", &reward_targets),
                ("alive_centered", &alive),
                ("row_weight", &weights),
            ] {
                self.train.set_input(name, values);
            }
            if c.context_motion_scale > 0. {
                let importance = (0..b)
                    .flat_map(|lane| {
                        history.motion_importance(lane, c.side, c.context_motion_scale)
                    })
                    .collect::<Vec<_>>();
                self.train.set_input("pixel_importance", &importance);
            }
            runtime::configure_d3_optimizer(
                &mut self.train,
                &c.optimizer(),
                self.updates + 1,
                c.learning_rate,
            );
            self.train.step();
            self.train.wait();
            for (i, value) in report[..4].iter_mut().enumerate() {
                let mut x = [0.];
                self.train.read_output_by_index(i, &mut x);
                if !x[0].is_finite() {
                    return Err("nonfinite denoising world loss");
                }
                *value += x[0];
            }
            let mut output = vec![0.; b * w];
            self.train.read_output_by_index(4, &mut output);
            if output.iter().any(|v| !v.is_finite()) {
                return Err("nonfinite training denoised frame");
            }
            let mut next = Vec::with_capacity(b * w);
            for lane in 0..b {
                let (_, cout, cskip, _) = conditioners(sigma[lane]);
                for i in 0..w {
                    let j = lane * w + i;
                    next.push(if incoming[lane] < 0 {
                        actual[j]
                    } else {
                        absolute(
                            cskip * noisy[j] + cout * output[j],
                            base[j],
                            c.residual_frame,
                        )
                    });
                }
            }
            history.push(&next, incoming);
            self.updates += 1;
            report[4] += 1.;
            report[5] += valid as f32;
            self.dirty = true;
        }
        let updates = report[4];
        if updates > 0. {
            for x in &mut report[..4] {
                *x /= updates;
            }
        }
        Ok(report)
    }
    fn denoise(
        &mut self,
        history: &History,
        actions: &[i32],
        x: &[f32],
        sigma: f32,
    ) -> Result<Vec<f32>, &'static str> {
        let c = &self.config;
        let b = c.batch_size;
        let w = 3 * c.side * c.side;
        let (cin, cout, cskip, cnoise) = conditioners(sigma);
        let mut input = Vec::new();
        for lane in 0..b {
            input.extend(history.condition(lane, actions[lane], c.action_count, true));
            input.extend(x[lane * w..(lane + 1) * w].iter().map(|v| v * cin));
            input.extend(std::iter::repeat_n(cnoise, c.side * c.side));
        }
        self.sampler.set_input("denoiser_input", &input);
        self.sampler.step();
        self.sampler.wait();
        let mut f = vec![0.; b * w];
        self.sampler.read_output_by_index(0, &mut f);
        if f.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite sampled denoised frame");
        }
        let base = history.base(actions, c.residual_frame);
        Ok(x.iter()
            .zip(f)
            .enumerate()
            .map(|(i, (&x, f))| absolute(cskip * x + cout * f, base[i], c.residual_frame))
            .collect())
    }
    pub fn predict(
        &mut self,
        images: &[f32],
        actions: &[i32],
        future: &[usize],
        seed: u64,
    ) -> Result<WorldPrediction, &'static str> {
        let c = self.config.clone();
        let b = c.batch_size;
        let w = 3 * c.side * c.side;
        if actions.len() < 5 * b
            || !actions.len().is_multiple_of(b)
            || images.len() != actions.len() * w
            || !valid_images(images)
            || actions
                .iter()
                .any(|&a| a < -1 || a >= c.action_count as i32)
            || future.is_empty()
            || !future.len().is_multiple_of(b)
            || future.len() > 64 * b
            || future.iter().any(|&a| a >= c.action_count)
        {
            return Err("invalid denoising forecast history or actions");
        }
        self.sync();
        let length = actions.len() / b;
        let mut history = History::empty(b, w);
        for t in 0..length - 1 {
            history.push(
                &images[t * b * w..(t + 1) * b * w]
                    .iter()
                    .map(|v| 2. * v)
                    .collect::<Vec<_>>(),
                &actions[t * b..(t + 1) * b],
            );
        }
        let mut observed_rng = StdRng::seed_from_u64(seed ^ 0x6f62736572766564);
        let noise = gaussian::normals(&mut observed_rng, b * w);
        let observed = images[(length - 1) * b * w..]
            .iter()
            .map(|v| 2. * v)
            .collect::<Vec<_>>();
        let base = history.base(&actions[(length - 1) * b..], c.residual_frame);
        let x = observed
            .iter()
            .zip(noise)
            .enumerate()
            .map(|(i, (&v, n))| coordinate(v, base[i], c.residual_frame) + 0.5 * n)
            .collect::<Vec<_>>();
        let posterior = self
            .denoise(&history, &actions[(length - 1) * b..], &x, 0.5)?
            .iter()
            .map(|v| v * 0.5)
            .collect();
        history.push(&observed, &actions[(length - 1) * b..]);
        let mut result = WorldPrediction {
            posterior,
            observations: Vec::new(),
            rewards: Vec::new(),
            reward_positive_probability: Vec::new(),
            continuations: Vec::new(),
            encoded: Vec::new(),
            posterior_features: Vec::new(),
            prior_features: Vec::new(),
        };
        let mut rng = StdRng::seed_from_u64(seed);
        let sigmas = schedule();
        let bins = TwoHotBins::new(c.value_bins);
        for a in future.chunks_exact(b) {
            let incoming = a.iter().map(|&a| a as i32).collect::<Vec<_>>();
            let base = history.base(&incoming, c.residual_frame);
            let mut x = gaussian::normals(&mut rng, b * w)
                .iter()
                .map(|v| v * SIGMA_MAX)
                .collect::<Vec<_>>();
            for pair in sigmas.windows(2) {
                let d = self.denoise(&history, &incoming, &x, pair[0])?;
                for (i, (v, pred)) in x.iter_mut().zip(d).enumerate() {
                    let pred = coordinate(pred, base[i], c.residual_frame);
                    *v += (*v - pred) / pair[0] * (pair[1] - pair[0]);
                }
            }
            for (i, v) in x.iter_mut().enumerate() {
                *v = absolute(*v, base[i], c.residual_frame);
            }
            let mut event_input = Vec::new();
            for (lane, &action) in incoming.iter().enumerate() {
                event_input.extend(history.condition(lane, action, c.action_count, false));
                event_input.extend_from_slice(&x[lane * w..(lane + 1) * w]);
            }
            self.events.set_input("event_input", &event_input);
            self.events.step();
            self.events.wait();
            let mut logits = vec![0.; b * c.value_bins];
            let mut cont = vec![0.; b];
            self.events.read_output_by_index(0, &mut logits);
            self.events.read_output_by_index(1, &mut cont);
            let mut scratch = vec![0.; c.value_bins];
            let mut reward = Vec::new();
            let mut positive = Vec::new();
            for row in logits.chunks_exact(c.value_bins) {
                reward.push(bins.decode_logits_with_scratch(row, &mut scratch));
                positive.push(
                    (scratch[c.value_bins / 2 + 1..].iter().sum::<f64>()
                        / scratch.iter().sum::<f64>()) as f32,
                );
            }
            let gamma = 1. - 1. / c.horizon as f32;
            let continuation = cont
                .iter()
                .map(|&v| gamma / (1. + (-v).exp()))
                .collect::<Vec<_>>();
            if reward
                .iter()
                .chain(&positive)
                .chain(&continuation)
                .any(|v| !v.is_finite())
            {
                return Err("nonfinite denoising event forecast");
            }
            result.rewards.push(reward);
            result.reward_positive_probability.push(positive);
            result.continuations.push(continuation);
            result.observations.extend(x.iter().map(|v| v * 0.5));
            history.push(&x, &incoming);
        }
        Ok(result)
    }
    pub fn save(&mut self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        if self.train.memory_summary().adam_state_bytes == 0 {
            if self.updates != 0 {
                return Err("missing trained denoising optimizer state".into());
            }
            runtime::configure_d3_optimizer(&mut self.train, &self.config.optimizer(), 0, 0.);
            self.train.clear_optimizer();
        }
        std::fs::create_dir(path)?;
        self.train
            .save_checkpoint(&path.join("world.safetensors"))?;
        std::fs::write(
            path.join("metadata.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"format":self.config.checkpoint_format(),"config":self.config,"updates":self.updates,"architecture":"conditional-unet16-32-64-history4","sampler":self.config.sampler_name(),"observed_reconstruction":"single-denoise-sigma0.5-not-rssm-posterior"}),
            )?,
        )?;
        Ok(())
    }
    pub fn restore(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let (config, updates) = read_header(&std::fs::read(path.join("metadata.json"))?)?;
        let mut model = Self::new(config)?;
        checkpoint::load_session(&mut model.train, &path.join("world.safetensors"))?;
        if u64::from(model.train.adam_step_count()) != updates {
            return Err("denoising optimizer age mismatch".into());
        }
        model.updates = updates;
        model.dirty = true;
        Ok(model)
    }
}

fn read_header(bytes: &[u8]) -> Result<(DenoisingConfig, u64), Box<dyn std::error::Error>> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Header {
        format: String,
        config: DenoisingConfig,
        updates: u64,
        architecture: String,
        sampler: String,
        observed_reconstruction: String,
    }
    let h: Header = serde_json::from_slice(bytes)?;
    if h.format != h.config.checkpoint_format()
        || h.architecture != "conditional-unet16-32-64-history4"
        || h.sampler != h.config.sampler_name()
        || h.observed_reconstruction != "single-denoise-sigma0.5-not-rssm-posterior"
        || h.updates > u64::from(u32::MAX)
    {
        return Err("invalid denoising checkpoint header".into());
    }
    h.config.validate()?;
    Ok((h.config, h.updates))
}
