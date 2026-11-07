use super::*;
use kindle::{DreamerCore, FeatureVectorAgent, FrameFlags, vision::Observation};

/// Saved-feature diagnostics and offline learning use the production core;
/// no separate world-model optimizer or CPU learning implementation.
#[pyclass(name = "FeatureCore", module = "kindle._native", unsendable)]
pub(crate) struct PyFeatureCore {
    inner: DreamerCore,
}

fn feature_bytes(bytes: &[u8]) -> PyResult<Observation> {
    if bytes.len() != 4 * Observation::LEN {
        return Err(PyValueError::new_err(
            "expected 3136 little-endian F32 features",
        ));
    }
    let values = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect::<Vec<_>>();
    if values.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("features must be finite"));
    }
    Ok(Observation::from_vec(values))
}

#[pymethods]
impl PyFeatureCore {
    #[new]
    fn new(config: &Bound<'_, PyAny>) -> PyResult<Self> {
        let encoded: String = config
            .py()
            .import("json")?
            .call_method1("dumps", (config,))?
            .extract()?;
        let config: DreamerConfig =
            serde_json::from_str(&encoded).map_err(|e| PyValueError::new_err(e.to_string()))?;
        config.check().map_err(PyValueError::new_err)?;
        if config.video_encoder.is_some()
            || config.observation_kind != kindle::ObservationKind::Features
        {
            return Err(PyValueError::new_err(
                "FeatureCore requires already encoded features",
            ));
        }
        Ok(Self {
            inner: DreamerCore::new(config).map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        })
    }

    #[classmethod]
    fn restore(_class: &Bound<'_, PyType>, checkpoint: &str) -> PyResult<Self> {
        let inner =
            DreamerCore::restore(checkpoint).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        if inner.config().video_encoder.is_some()
            || inner.config().observation_kind != kindle::ObservationKind::Features
        {
            return Err(PyValueError::new_err("expected a saved-feature core"));
        }
        Ok(Self { inner })
    }

    #[getter]
    fn latent_feature(&self) -> Vec<f32> {
        self.inner.latent_feature().to_vec()
    }

    #[getter]
    fn encoded_observation(&self) -> Vec<f32> {
        self.inner.encoded_observation().to_vec()
    }

    #[getter]
    fn deterministic_size(&self) -> usize {
        self.inner.config().network().deter
    }

    fn begin_episode(&mut self, features: &[u8]) -> PyResult<()> {
        self.inner.begin_episode(feature_bytes(features)?);
        Ok(())
    }

    fn observe(
        &mut self,
        action: usize,
        features: &[u8],
        reward: f32,
        terminated: bool,
        truncated: bool,
    ) -> PyResult<()> {
        let observation = feature_bytes(features)?;
        if action >= self.inner.config().action_count || !reward.is_finite() {
            return Err(PyValueError::new_err("invalid recorded action or reward"));
        }
        self.inner.observe_recorded(
            action,
            observation,
            Reward {
                extrinsic: reward,
                intrinsic: 0.0,
            },
            FrameFlags {
                is_first: false,
                is_last: terminated || truncated,
                is_terminal: terminated,
            },
        );
        Ok(())
    }

    fn learn<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_python(py, &self.inner.learn())
    }

    fn forecast(&mut self, actions: Vec<usize>) -> PyResult<(Vec<f32>, Vec<Vec<f32>>)> {
        if actions.is_empty()
            || actions
                .iter()
                .any(|a| *a >= self.inner.config().action_count)
        {
            return Err(PyValueError::new_err(
                "nonempty valid action sequence required",
            ));
        }
        let (rewards, observations) = self.inner.prior_diagnostic_rollout(&actions);
        let last = actions.len() - 1;
        // The offline screen only scores h1 and the final horizon. Avoid
        // materializing every intermediate feature map as Python floats.
        Ok((
            vec![rewards[0], rewards[last]],
            vec![observations[0].clone(), observations[last].clone()],
        ))
    }

    /// Read-only first/final prior belief and predicted-feature endpoints.
    fn forecast_states(&mut self, actions: Vec<usize>) -> PyResult<ForecastEndpoints> {
        if actions.is_empty()
            || actions
                .iter()
                .any(|a| *a >= self.inner.config().action_count)
        {
            return Err(PyValueError::new_err(
                "nonempty valid action sequence required",
            ));
        }
        let (states, observations) = self.inner.prior_state_rollout(&actions);
        let last = actions.len() - 1;
        Ok((
            vec![states[0].clone(), states[last].clone()],
            vec![observations[0].clone(), observations[last].clone()],
        ))
    }

    fn save_checkpoint(&mut self, path: &str) -> PyResult<()> {
        self.inner
            .save_checkpoint(path)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[getter]
    fn learner_step(&self) -> u64 {
        self.inner.learner_step()
    }
    #[getter]
    fn environment_step(&self) -> u64 {
        self.inner.environment_step()
    }
    #[getter]
    fn replay_len(&self) -> usize {
        self.inner.replay_len()
    }
    #[getter]
    fn gpu_device<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_python(py, &self.inner.gpu_device())
    }
    #[getter]
    fn gpu_memory_budget<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_python(py, &self.inner.gpu_memory_budget())
    }
}

#[pyclass(name = "FeatureVectorAgent", module = "kindle", unsendable)]
pub(crate) struct PyFeatureVectorAgent {
    inner: FeatureVectorAgent,
}

fn observations(values: Vec<Vec<f32>>) -> PyResult<Vec<Observation>> {
    if values
        .iter()
        .any(|v| v.len() != Observation::LEN || v.iter().any(|x| !x.is_finite()))
    {
        return Err(PyValueError::new_err(
            "each observation must contain 3136 finite float values",
        ));
    }
    Ok(values.into_iter().map(Observation::from_vec).collect())
}

#[pymethods]
impl PyFeatureVectorAgent {
    #[new]
    fn new(num_envs: usize, config: &Bound<'_, PyAny>) -> PyResult<Self> {
        let encoded: String = config
            .py()
            .import("json")?
            .call_method1("dumps", (config,))?
            .extract()?;
        let config: DreamerConfig =
            serde_json::from_str(&encoded).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let inner = FeatureVectorAgent::new(config, num_envs)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn begin_episodes(&mut self, streams: Vec<usize>, features: Vec<Vec<f32>>) -> PyResult<()> {
        vector::validate_streams(&streams, self.inner.stream_count(), features.len())?;
        self.inner
            .begin_episodes(streams.into_iter().zip(observations(features)?).collect());
        Ok(())
    }

    #[pyo3(signature = (greedy = false))]
    fn act(&mut self, greedy: bool) -> Vec<usize> {
        self.inner.act(if greedy {
            ActionMode::Greedy
        } else {
            ActionMode::Sample
        })
    }

    fn observe(
        &mut self,
        streams: Vec<usize>,
        features: Vec<Vec<f32>>,
        rewards: Vec<f32>,
        terminated: Vec<bool>,
        truncated: Vec<bool>,
    ) -> PyResult<Vec<(f32, f32)>> {
        vector::validate_streams(&streams, self.inner.stream_count(), features.len())?;
        let count = features.len();
        if rewards.len() != count
            || terminated.len() != count
            || truncated.len() != count
            || rewards.iter().any(|x| !x.is_finite())
        {
            return Err(PyValueError::new_err(
                "transition fields must have matching lengths and finite rewards",
            ));
        }
        let arrivals = streams
            .into_iter()
            .zip(observations(features)?)
            .enumerate()
            .map(|(row, (id, obs))| {
                (
                    id,
                    obs,
                    FrameFlags {
                        is_first: false,
                        is_last: terminated[row] || truncated[row],
                        is_terminal: terminated[row],
                    },
                    Reward {
                        extrinsic: rewards[row],
                        intrinsic: 0.0,
                    },
                )
            })
            .collect();
        Ok(self
            .inner
            .observe(arrivals)
            .into_iter()
            .map(|r| (r.extrinsic, r.intrinsic))
            .collect())
    }

    #[pyo3(signature = (maximum_updates = usize::MAX))]
    fn learn_scheduled<'py>(
        &mut self,
        py: Python<'py>,
        maximum_updates: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        reports_to_python(py, &self.inner.learn_scheduled(maximum_updates))
    }

    fn save_checkpoint(&mut self, path: &str) -> PyResult<()> {
        self.inner
            .save_checkpoint(path)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[getter]
    fn config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_python(py, self.inner.learner().config())
    }
    #[getter]
    fn provenance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_python(py, &self.inner.learner().provenance())
    }
    #[getter]
    fn gpu_device<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_python(py, &self.inner.learner().gpu_device())
    }
    #[getter]
    fn gpu_memory_budget<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_python(py, &self.inner.learner().gpu_memory_budget())
    }
    #[getter]
    fn trainable_parameter_counts(&self) -> (usize, usize) {
        self.inner.learner().trainable_parameter_counts()
    }
    #[getter]
    fn learner_step(&self) -> u64 {
        self.inner.learner().learner_step()
    }
    #[getter]
    fn environment_step(&self) -> u64 {
        self.inner.learner().environment_step()
    }
    #[getter]
    fn training_debt(&self) -> f32 {
        self.inner.training_debt()
    }
    #[getter]
    fn num_envs(&self) -> usize {
        self.inner.stream_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn feature_input_validation_precedes_native_mutation() {
        assert!(observations(vec![vec![0.0; Observation::LEN]]).is_ok());
        assert!(observations(vec![vec![0.0; Observation::LEN - 1]]).is_err());
        assert!(observations(vec![vec![f32::NAN; Observation::LEN]]).is_err());
        assert!(feature_bytes(&vec![0; 4 * Observation::LEN]).is_ok());
        assert!(feature_bytes(&[0; 4]).is_err());
        assert!(feature_bytes(&f32::NAN.to_le_bytes().repeat(Observation::LEN)).is_err());
    }
}
