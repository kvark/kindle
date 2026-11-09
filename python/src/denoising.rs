use super::world::{float_bytes, floats};
use super::*;
use kindle::dreamer::{DenoisingConfig, DenoisingWorldLearner};

#[pyclass(name = "DenoisingWorldLearner", module = "kindle._native", unsendable)]
pub(crate) struct PyDenoisingWorldLearner {
    inner: DenoisingWorldLearner,
}
#[pymethods]
impl PyDenoisingWorldLearner {
    #[new]
    fn new(config: &Bound<'_, PyAny>) -> PyResult<Self> {
        let encoded: String = config
            .py()
            .import("json")?
            .call_method1("dumps", (config,))?
            .extract()?;
        let config: DenoisingConfig =
            serde_json::from_str(&encoded).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: DenoisingWorldLearner::new(config)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        })
    }
    #[classmethod]
    fn restore(_class: &Bound<'_, PyType>, checkpoint: &str) -> PyResult<Self> {
        Ok(Self {
            inner: DenoisingWorldLearner::restore(Path::new(checkpoint))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        })
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
    fn gpu_device<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_python(py, &self.inner.gpu_device())
    }
    #[getter]
    fn gpu_memory_budget<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_python(py, &self.inner.gpu_memory_budget())
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
        float_bytes(
            py,
            &self
                .inner
                .preprocess(&frames)
                .map_err(PyValueError::new_err)?,
        )
    }
    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names()
    }
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
    /// Observed reconstruction is noisy-frame denoising at sigma .5, not an RSSM posterior.
    #[pyo3(signature=(observations,previous_actions,future_actions,*,seed,include_features=false))]
    fn predict<'py>(
        &mut self,
        py: Python<'py>,
        observations: &[u8],
        previous_actions: Vec<i32>,
        future_actions: Vec<usize>,
        seed: u64,
        include_features: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        if include_features {
            return Err(PyValueError::new_err(
                "pixel history has no learned RSSM features",
            ));
        }
        let r = self
            .inner
            .predict(
                &floats(observations)?,
                &previous_actions,
                &future_actions,
                seed,
            )
            .map_err(PyValueError::new_err)?;
        let out = pyo3::types::PyDict::new(py);
        out.set_item("posterior_rgb", float_bytes(py, &r.posterior)?)?;
        out.set_item(
            "observed_reconstruction",
            "single-denoise-sigma0.5-not-rssm-posterior",
        )?;
        out.set_item("prior_rgb", float_bytes(py, &r.observations)?)?;
        out.set_item("reward", r.rewards)?;
        out.set_item("reward_positive_probability", r.reward_positive_probability)?;
        out.set_item("continuation", r.continuations)?;
        Ok(out.into_any())
    }
    fn save(&mut self, checkpoint: &str) -> PyResult<()> {
        self.inner
            .save(Path::new(checkpoint))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}
