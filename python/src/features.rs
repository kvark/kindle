use super::*;
use kindle::{FeatureVectorAgent, FrameFlags, vision::Observation};

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
    }
}
