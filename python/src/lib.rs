//! Python bindings for the pixel-first Dreamer baseline.

use std::path::Path;

use kindle::vision::{DinoPerception, OBSERVATION_CHANNELS, OBSERVATION_GRID};
use kindle::{
    ActionMode, DreamerAgent, DreamerConfig, LearnReport, ModelSize, Reward, RgbFrame, Transition,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyModule, PyType};

#[pyclass(name = "Agent", module = "kindle", unsendable)]
struct PyAgent {
    inner: DreamerAgent,
}

/// Frozen perception-only session for representation probes.
///
/// The first element returned by `encode` is the projected 14x14 patch grid;
/// the second is the exact pooled 7x7 observation consumed by Dreamer.
#[pyclass(name = "DinoPerception", module = "kindle._native", unsendable)]
struct PyDinoPerception {
    inner: DinoPerception,
}

#[pymethods]
impl PyDinoPerception {
    #[new]
    #[pyo3(signature = (dino_checkpoint, dino_plan_cache = None))]
    fn new(dino_checkpoint: &str, dino_plan_cache: Option<&str>) -> PyResult<Self> {
        let cache = dino_plan_cache.map(Path::new);
        let inner = DinoPerception::load_vits16(dino_checkpoint, None, cache)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(Self { inner })
    }

    fn encode(&mut self, frame: &Bound<'_, PyAny>) -> PyResult<(Vec<f32>, Vec<f32>)> {
        let frame = parse_rgb_frame(frame)?;
        let pooled = self
            .inner
            .encode_frame_rgb8(frame.pixels(), frame.width(), frame.height());
        Ok((
            self.inner.projected_patches().to_vec(),
            pooled.as_slice().to_vec(),
        ))
    }

    #[getter]
    fn projected_shape(&self) -> (usize, usize, usize) {
        let grid = self.inner.projected_grid();
        (grid, grid, OBSERVATION_CHANNELS)
    }

    #[getter]
    fn pooled_shape(&self) -> (usize, usize, usize) {
        (OBSERVATION_GRID, OBSERVATION_GRID, OBSERVATION_CHANNELS)
    }

    #[getter]
    fn gpu_device<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_python(py, &self.inner.gpu_device())
    }
}

#[pymethods]
impl PyAgent {
    #[new]
    #[pyo3(signature = (
        dino_checkpoint,
        num_actions,
        model_size = "12m",
        observation_decoder_depth = None,
        seed = 0,
        replay_capacity = None,
        batch_size = None,
        batch_length = None,
        world_backprop_length = None,
        world_microbatch_size = None,
        train_ratio = None,
        imagination_length = None,
        intrinsic_reward_scale = 0.0,
        extrinsic_reward_scale = 1.0,
        visitation_bonus = false,
        learning_rate = None,
        behavior_learning_rate = None,
        actor_learning_starts = None,
        learning_rate_warmup = None,
        free_nats = None,
        dynamics_free_nats = None,
        dynamics_loss_scale = None,
        reconstruction_loss_scale = None,
        future_prediction_loss_scale = None,
        agc = None,
        replay_value_gradient = None,
        dino_plan_cache = None,
        skip_full_optimize = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        dino_checkpoint: &str,
        num_actions: usize,
        model_size: &str,
        observation_decoder_depth: Option<usize>,
        seed: u64,
        replay_capacity: Option<usize>,
        batch_size: Option<usize>,
        batch_length: Option<usize>,
        world_backprop_length: Option<usize>,
        world_microbatch_size: Option<usize>,
        train_ratio: Option<f32>,
        imagination_length: Option<usize>,
        intrinsic_reward_scale: f32,
        extrinsic_reward_scale: f32,
        visitation_bonus: bool,
        learning_rate: Option<f32>,
        behavior_learning_rate: Option<f32>,
        actor_learning_starts: Option<u64>,
        learning_rate_warmup: Option<u64>,
        free_nats: Option<f32>,
        dynamics_free_nats: Option<f32>,
        dynamics_loss_scale: Option<f32>,
        reconstruction_loss_scale: Option<f32>,
        future_prediction_loss_scale: Option<f32>,
        agc: Option<f32>,
        replay_value_gradient: Option<bool>,
        dino_plan_cache: Option<&str>,
        skip_full_optimize: bool,
    ) -> PyResult<Self> {
        validate_action_count(num_actions)?;
        let mut config = DreamerConfig::new(num_actions);
        config.model_size = parse_model_size(model_size)?;
        if let Some(value) = observation_decoder_depth {
            config.observation_decoder_depth = value;
        }
        config.seed = seed;
        config.intrinsic_reward_scale = intrinsic_reward_scale;
        config.extrinsic_reward_scale = extrinsic_reward_scale;
        config.visitation_bonus = visitation_bonus;
        config.skip_full_optimize = skip_full_optimize;
        if let Some(value) = replay_capacity {
            config.replay_capacity = value;
        }
        if let Some(value) = batch_size {
            config.batch_size = value;
        }
        if let Some(value) = batch_length {
            config.batch_length = value;
        }
        if let Some(value) = world_backprop_length {
            config.world_backprop_length = value;
        }
        if let Some(value) = world_microbatch_size {
            config.world_microbatch_size = Some(value);
        }
        if let Some(value) = train_ratio {
            config.train_ratio = value;
        }
        if let Some(value) = imagination_length {
            config.imagination_length = value;
        }
        if let Some(value) = learning_rate {
            config.learning_rate = value;
        }
        if let Some(value) = behavior_learning_rate {
            config.behavior_learning_rate = Some(value);
        }
        if let Some(value) = actor_learning_starts {
            config.actor_learning_starts = value;
        }
        if let Some(value) = learning_rate_warmup {
            config.learning_rate_warmup = value;
        }
        if let Some(value) = free_nats {
            config.free_nats = value;
        }
        if let Some(value) = dynamics_free_nats {
            config.dynamics_free_nats = Some(value);
        }
        if let Some(value) = dynamics_loss_scale {
            config.loss_scales.dynamics = value;
        }
        if let Some(value) = reconstruction_loss_scale {
            config.loss_scales.reconstruction = value;
        }
        if let Some(value) = future_prediction_loss_scale {
            config.loss_scales.future_prediction = value;
        }
        if let Some(value) = agc {
            config.agc = value;
        }
        if let Some(value) = replay_value_gradient {
            config.replay_value_gradient = value;
        }
        config
            .check()
            .map_err(|error| PyValueError::new_err(format!("invalid Dreamer config: {error}")))?;
        let cache = dino_plan_cache.map(Path::new);
        let inner = DreamerAgent::new(config, dino_checkpoint, cache)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(Self { inner })
    }

    #[classmethod]
    #[pyo3(signature = (dreamer_checkpoint, dino_checkpoint, dino_plan_cache = None))]
    fn restore(
        _class: &Bound<'_, PyType>,
        dreamer_checkpoint: &str,
        dino_checkpoint: &str,
        dino_plan_cache: Option<&str>,
    ) -> PyResult<Self> {
        let cache = dino_plan_cache.map(Path::new);
        let inner = DreamerAgent::restore(dreamer_checkpoint, dino_checkpoint, cache)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(Self { inner })
    }

    /// Supply the first H×W×3 uint8 frame after reset.
    fn begin_episode(&mut self, frame: &Bound<'_, PyAny>) -> PyResult<()> {
        let frame = parse_rgb_frame(frame)?;
        self.inner.begin_episode(&frame);
        Ok(())
    }

    #[pyo3(signature = (greedy = false, action_mask = None))]
    fn act(&mut self, greedy: bool, action_mask: Option<Vec<bool>>) -> PyResult<usize> {
        if let Some(mask) = &action_mask {
            let expected = self.inner.core().config().action_count;
            if mask.len() != expected {
                return Err(PyValueError::new_err(format!(
                    "action_mask has {} entries, expected {expected}",
                    mask.len()
                )));
            }
            if !mask.iter().any(|valid| *valid) {
                return Err(PyValueError::new_err(
                    "action_mask must enable at least one action",
                ));
            }
        }
        let mode = if greedy {
            ActionMode::Greedy
        } else {
            ActionMode::Sample
        };
        Ok(self.inner.act(mode, action_mask.as_deref()))
    }

    /// Reward predicted from the current posterior state.
    ///
    /// This diagnostic does not change recurrent state or random streams.
    fn posterior_reward_prediction(&mut self) -> f32 {
        self.inner.posterior_reward_prediction()
    }

    /// Value predicted from the current posterior state.
    ///
    /// This diagnostic does not change recurrent state or random streams.
    fn posterior_value_prediction(&mut self) -> f32 {
        self.inner.posterior_value_prediction()
    }

    /// Current action probabilities after categorical unimix.
    ///
    /// This diagnostic does not sample an action or change recurrent state.
    fn posterior_action_probabilities(&mut self) -> Vec<f32> {
        self.inner.posterior_action_probabilities(None)
    }

    /// Current frozen-DINO observation consumed by the world model.
    ///
    /// The returned copy is read-only and does not change recurrent state.
    #[getter]
    fn dino_observation(&self) -> Vec<f32> {
        self.inner.dino_observation().to_vec()
    }

    /// Frozen-DINO observation reconstructed from the current posterior.
    ///
    /// This diagnostic does not change recurrent state or random streams.
    fn observation_prediction(&mut self) -> Vec<f32> {
        self.inner.observation_prediction()
    }

    /// Reward predicted after applying one action to the current latent state.
    ///
    /// This diagnostic does not change recurrent state or random streams.
    fn prior_reward_prediction(&mut self, action: usize) -> PyResult<f32> {
        let action_count = self.inner.core().config().action_count;
        if action >= action_count {
            return Err(PyValueError::new_err(format!(
                "action {action} is out of range for {action_count} actions"
            )));
        }
        Ok(self.inner.prior_reward_prediction(action))
    }

    /// Open-loop prior rewards and decoded frozen-DINO observations.
    ///
    /// This diagnostic clones its categorical random stream and leaves the
    /// live posterior, policy, and all training state unchanged.
    fn prior_diagnostic_rollout(
        &mut self,
        actions: Vec<usize>,
    ) -> PyResult<(Vec<f32>, Vec<Vec<f32>>)> {
        if actions.is_empty() {
            return Err(PyValueError::new_err("actions must not be empty"));
        }
        let action_count = self.inner.core().config().action_count;
        if let Some(action) = actions.iter().find(|action| **action >= action_count) {
            return Err(PyValueError::new_err(format!(
                "action {action} is out of range for {action_count} actions"
            )));
        }
        Ok(self.inner.prior_diagnostic_rollout(&actions))
    }

    /// Open-loop reward, continuation, and value predictions.
    ///
    /// This diagnostic clones its categorical random stream and leaves the
    /// live posterior, policy, and all training state unchanged.
    fn prior_behavior_rollout(
        &mut self,
        actions: Vec<usize>,
    ) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        if actions.is_empty() {
            return Err(PyValueError::new_err("actions must not be empty"));
        }
        let action_count = self.inner.core().config().action_count;
        if let Some(action) = actions.iter().find(|action| **action >= action_count) {
            return Err(PyValueError::new_err(format!(
                "action {action} is out of range for {action_count} actions"
            )));
        }
        Ok(self.inner.prior_behavior_rollout(&actions))
    }

    /// Record an arrival and return (extrinsic, intrinsic), including configured
    /// novelty. These channels are unscaled, as stored in replay.
    #[pyo3(signature = (
        frame,
        extrinsic_reward = 0.0,
        intrinsic_reward = 0.0,
        terminated = false,
        truncated = false,
    ))]
    fn observe(
        &mut self,
        frame: &Bound<'_, PyAny>,
        extrinsic_reward: f32,
        intrinsic_reward: f32,
        terminated: bool,
        truncated: bool,
    ) -> PyResult<(f32, f32)> {
        if !extrinsic_reward.is_finite() || !intrinsic_reward.is_finite() {
            return Err(PyValueError::new_err("rewards must be finite"));
        }
        let transition = Transition {
            frame: parse_rgb_frame(frame)?,
            reward: Reward {
                extrinsic: extrinsic_reward,
                intrinsic: intrinsic_reward,
            },
            terminated,
            truncated,
        };
        let reward = self.inner.observe(&transition);
        Ok((reward.extrinsic, reward.intrinsic))
    }

    #[pyo3(signature = (updates = 1))]
    fn learn<'py>(&mut self, py: Python<'py>, updates: usize) -> PyResult<Bound<'py, PyAny>> {
        let reports = (0..updates)
            .filter_map(|_| self.inner.learn())
            .collect::<Vec<_>>();
        reports_to_python(py, &reports)
    }

    #[pyo3(signature = (maximum_updates = 1))]
    fn learn_scheduled<'py>(
        &mut self,
        py: Python<'py>,
        maximum_updates: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let reports = self.inner.learn_scheduled(maximum_updates);
        reports_to_python(py, &reports)
    }

    fn save_checkpoint(&mut self, path: &str) -> PyResult<()> {
        self.inner
            .save_checkpoint(path)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    #[getter]
    fn learner_step(&self) -> u64 {
        self.inner.core().learner_step()
    }

    #[getter]
    fn environment_step(&self) -> u64 {
        self.inner.core().environment_step()
    }

    #[getter]
    fn replay_len(&self) -> usize {
        self.inner.core().replay_len()
    }

    #[getter]
    fn trainable_parameter_counts(&self) -> (usize, usize) {
        self.inner.core().trainable_parameter_counts()
    }

    #[getter]
    fn gpu_device<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_python(py, &self.inner.core().gpu_device())
    }

    #[getter]
    fn config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_python(py, self.inner.core().config())
    }
}

#[pyfunction]
#[pyo3(signature = (num_actions, model_size = "12m"))]
fn default_config<'py>(
    py: Python<'py>,
    num_actions: usize,
    model_size: &str,
) -> PyResult<Bound<'py, PyAny>> {
    validate_action_count(num_actions)?;
    let mut config = DreamerConfig::new(num_actions);
    config.model_size = parse_model_size(model_size)?;
    json_to_python(py, &config)
}

fn validate_action_count(num_actions: usize) -> PyResult<()> {
    if num_actions <= 1 {
        Err(PyValueError::new_err(
            "num_actions must be greater than one",
        ))
    } else {
        Ok(())
    }
}

fn parse_model_size(value: &str) -> PyResult<ModelSize> {
    match value {
        "tiny" => Ok(ModelSize::Tiny),
        "1m" => Ok(ModelSize::Size1M),
        "12m" => Ok(ModelSize::Size12M),
        "25m" => Ok(ModelSize::Size25M),
        "50m" => Ok(ModelSize::Size50M),
        "100m" => Ok(ModelSize::Size100M),
        "200m" => Ok(ModelSize::Size200M),
        _ => Err(PyValueError::new_err(format!(
            "unknown model_size {value:?}; expected tiny, 1m, 12m, 25m, 50m, 100m, or 200m"
        ))),
    }
}

fn parse_rgb_frame(frame: &Bound<'_, PyAny>) -> PyResult<RgbFrame> {
    let shape = frame
        .getattr("shape")
        .map_err(|_| PyValueError::new_err("frame must expose shape (height, width, 3)"))?;
    let (height, width, channels): (usize, usize, usize) = shape
        .extract()
        .map_err(|_| PyValueError::new_err("frame shape must be (height, width, 3)"))?;
    if height == 0 || width == 0 || channels != 3 {
        return Err(PyValueError::new_err(format!(
            "frame shape must be non-empty H×W×3, got {height}×{width}×{channels}"
        )));
    }
    let bytes = frame
        .call_method0("tobytes")
        .map_err(|_| PyValueError::new_err("frame must provide tobytes()"))?;
    let bytes = bytes
        .cast::<PyBytes>()
        .map_err(|_| PyValueError::new_err("frame.tobytes() must return bytes"))?;
    let bytes = bytes.as_bytes();
    let expected = height
        .checked_mul(width)
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or_else(|| PyValueError::new_err("frame dimensions are too large"))?;
    if bytes.len() != expected {
        return Err(PyValueError::new_err(format!(
            "frame must use uint8 RGB storage: got {} bytes for shape {height}×{width}×3, expected {expected}",
            bytes.len()
        )));
    }
    Ok(RgbFrame::new(width, height, bytes.to_vec()))
}

fn reports_to_python<'py>(py: Python<'py>, reports: &[LearnReport]) -> PyResult<Bound<'py, PyAny>> {
    json_to_python(py, reports)
}

fn json_to_python<'py, T: serde::Serialize + ?Sized>(
    py: Python<'py>,
    value: &T,
) -> PyResult<Bound<'py, PyAny>> {
    let encoded =
        serde_json::to_string(value).map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    py.import("json")?.call_method1("loads", (encoded,))
}

#[pymodule]
fn _native(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyAgent>()?;
    module.add_class::<PyDinoPerception>()?;
    module.add_function(wrap_pyfunction!(default_config, module)?)?;
    module.add("DINO_MODEL_ID", kindle::vision::VITS16_MODEL_ID)?;
    module.add(
        "DINO_CHECKPOINT_REVISION",
        kindle::vision::VITS16_CHECKPOINT_REV,
    )?;
    module.add(
        "DREAMERV3_REVISION",
        kindle::dreamer::DREAMERV3_UPSTREAM_REV,
    )?;
    Ok(())
}
