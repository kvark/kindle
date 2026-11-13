use super::*;
use kindle::dreamer::{PixelLoss, WorldModelLearner};

#[pyclass(name = "WorldModelLearner", module = "kindle._native", unsendable)]
pub(crate) struct PyWorldModelLearner {
    inner: WorldModelLearner,
}

pub(super) fn floats(bytes: &[u8]) -> PyResult<Vec<f32>> {
    let (words, remainder) = bytes.as_chunks::<4>();
    if !remainder.is_empty() {
        return Err(PyValueError::new_err("expected little-endian F32 bytes"));
    }
    Ok(words.iter().map(|&word| f32::from_le_bytes(word)).collect())
}

pub(super) fn float_bytes<'py>(py: Python<'py>, values: &[f32]) -> PyResult<Bound<'py, PyBytes>> {
    PyBytes::new_with(py, values.len() * 4, |output| {
        for (word, value) in output.as_chunks_mut::<4>().0.iter_mut().zip(values) {
            word.copy_from_slice(&value.to_le_bytes());
        }
        Ok(())
    })
}

#[pymethods]
impl PyWorldModelLearner {
    #[new]
    #[pyo3(signature = (config, *, prediction_length=0, pixel_loss="squared_error"))]
    fn new(
        config: &Bound<'_, PyAny>,
        prediction_length: usize,
        pixel_loss: &str,
    ) -> PyResult<Self> {
        let pixel_loss: PixelLoss =
            serde_json::from_value(serde_json::Value::String(pixel_loss.into()))
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let encoded: String = config
            .py()
            .import("json")?
            .call_method1("dumps", (config,))?
            .extract()?;
        let config: DreamerConfig =
            serde_json::from_str(&encoded).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: WorldModelLearner::with_pixel_loss(config, prediction_length, pixel_loss)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        })
    }

    #[classmethod]
    fn restore(_class: &Bound<'_, PyType>, checkpoint: &str) -> PyResult<Self> {
        Ok(Self {
            inner: WorldModelLearner::restore(Path::new(checkpoint))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        })
    }

    #[classmethod]
    fn start_frozen_visual_phase(_class: &Bound<'_, PyType>, checkpoint: &str) -> PyResult<Self> {
        Ok(Self {
            inner: WorldModelLearner::start_frozen_visual_phase(Path::new(checkpoint))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        })
    }

    #[classmethod]
    fn start_fixed_teacher_phase(_class: &Bound<'_, PyType>, checkpoint: &str) -> PyResult<Self> {
        Ok(Self {
            inner: WorldModelLearner::start_fixed_teacher_phase(Path::new(checkpoint))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        })
    }

    #[classmethod]
    fn start_fixed_teacher_events_phase(
        _class: &Bound<'_, PyType>,
        checkpoint: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: WorldModelLearner::start_fixed_teacher_events_phase(Path::new(checkpoint))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        })
    }

    #[classmethod]
    fn start_fixed_teacher_image_phase(
        _class: &Bound<'_, PyType>,
        checkpoint: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: WorldModelLearner::start_fixed_teacher_image_phase(Path::new(checkpoint))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        })
    }

    fn fixed_teacher_targets<'py>(
        &mut self,
        py: Python<'py>,
        observations: &[u8],
        previous_actions: Vec<i32>,
        sampling_seeds: Vec<u64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (h, z) = self
            .inner
            .fixed_teacher_targets(&floats(observations)?, &previous_actions, &sampling_seeds)
            .map_err(PyValueError::new_err)?;
        let output = pyo3::types::PyDict::new(py);
        output.set_item("deter", float_bytes(py, &h)?)?;
        output.set_item("stoch", float_bytes(py, &z)?)?;
        output.set_item("privileged_training_targets", true)?;
        Ok(output.into_any())
    }

    #[getter]
    fn config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_python(py, self.inner.config())
    }

    #[getter]
    fn updates(&self) -> u64 {
        self.inner.updates()
    }

    #[getter]
    fn training_scope(&self) -> &'static str {
        self.inner.training_scope()
    }

    #[getter]
    fn phase_start_updates(&self) -> Option<u64> {
        self.inner.phase_start_updates()
    }

    #[getter]
    fn prediction_length(&self) -> usize {
        self.inner.prediction_length()
    }

    #[getter]
    fn pixel_loss<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_python(py, &self.inner.pixel_loss())
    }

    #[getter]
    fn feedback_sharpness(&self) -> f32 {
        self.inner.feedback_sharpness()
    }

    #[setter]
    fn set_feedback_sharpness(&mut self, temperature: f32) -> PyResult<()> {
        self.inner
            .set_feedback_sharpness(temperature)
            .map_err(PyValueError::new_err)
    }

    #[getter]
    fn gpu_device<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_python(py, &self.inner.gpu_device())
    }

    #[getter]
    fn gpu_memory_budget<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_python(py, &self.inner.gpu_memory_budget())
    }

    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names()
    }

    fn preprocess<'py>(
        &mut self,
        py: Python<'py>,
        frames: Vec<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let frames = frames
            .iter()
            .map(parse_rgb_frame)
            .collect::<PyResult<Vec<_>>>()?;
        let pixels = self
            .inner
            .preprocess(&frames)
            .map_err(PyValueError::new_err)?;
        float_bytes(py, &pixels)
    }

    /// Centered CHW images and arrival-aligned actions are time-major then batch.
    /// -1 clears history. Mid-window episode resets have zero reward and are nonterminal;
    /// a clipped first arrival may retain its known incoming reward/terminal target.
    fn learn(
        &mut self,
        observations: &[u8],
        previous_actions: Vec<i32>,
        rewards: Vec<f32>,
        terminals: Vec<bool>,
    ) -> PyResult<Vec<f32>> {
        self.inner
            .learn(
                &floats(observations)?,
                &previous_actions,
                &rewards,
                &terminals,
            )
            .map_err(PyValueError::new_err)
    }

    /// Returns posterior RGB bytes, future RGB bytes, rewards and discounted continuation.
    #[pyo3(signature = (observations, previous_actions, future_actions, *, seed, include_features=false))]
    fn predict<'py>(
        &mut self,
        py: Python<'py>,
        observations: &[u8],
        previous_actions: Vec<i32>,
        future_actions: Vec<usize>,
        seed: u64,
        include_features: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let result = self
            .inner
            .predict_with_features(
                &floats(observations)?,
                &previous_actions,
                &future_actions,
                seed,
                include_features,
            )
            .map_err(PyValueError::new_err)?;
        let output = pyo3::types::PyDict::new(py);
        output.set_item("posterior_rgb", float_bytes(py, &result.posterior)?)?;
        output.set_item("prior_rgb", float_bytes(py, &result.observations)?)?;
        output.set_item("reward", result.rewards)?;
        output.set_item(
            "reward_positive_probability",
            result.reward_positive_probability,
        )?;
        output.set_item("continuation", result.continuations)?;
        if include_features {
            output.set_item("encoder_features", float_bytes(py, &result.encoded)?)?;
            output.set_item(
                "posterior_features",
                float_bytes(py, &result.posterior_features)?,
            )?;
            output.set_item("prior_features", float_bytes(py, &result.prior_features)?)?;
        }
        Ok(output.into_any())
    }

    fn save(&mut self, checkpoint: &str) -> PyResult<()> {
        self.inner
            .save(Path::new(checkpoint))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}
