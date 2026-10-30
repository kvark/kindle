use super::*;
use kindle::vision::probe::RegressionProbe;
use kindle::vision::reconstruction::ReconstructionEncoder;

#[pyclass(name = "RegressionProbe", module = "kindle._native", unsendable)]
pub(crate) struct PyRegressionProbe {
    inner: RegressionProbe,
}

#[pyclass(name = "ReconstructionEncoder", module = "kindle._native", unsendable)]
pub(crate) struct PyReconstructionEncoder {
    inner: ReconstructionEncoder,
}

#[pymethods]
impl PyReconstructionEncoder {
    #[getter]
    fn architecture(&self) -> &'static str {
        "cnn"
    }

    #[getter]
    fn patch_token_shape(&self) -> (usize, usize, usize, usize) {
        (
            self.inner.batch_size(),
            14,
            14,
            kindle::vision::reconstruction::CHANNELS,
        )
    }

    #[new]
    #[pyo3(signature = (*, batch = 16, seed = 2709))]
    fn new(batch: usize, seed: u64) -> PyResult<Self> {
        Ok(Self {
            inner: ReconstructionEncoder::new(batch, seed)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    #[pyo3(signature = (frames, *, learning_rate = None))]
    fn process(
        &mut self,
        frames: Vec<Bound<'_, PyAny>>,
        learning_rate: Option<f32>,
    ) -> PyResult<f32> {
        if frames.len() != self.inner.batch_size() {
            return Err(PyValueError::new_err(
                "complete reconstruction batch required",
            ));
        }
        let frames = frames
            .iter()
            .map(parse_rgb_frame)
            .collect::<PyResult<Vec<_>>>()?;
        self.inner
            .process(&frames, learning_rate)
            .map_err(PyValueError::new_err)
    }

    fn patch_tokens<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let values = self.inner.tokens();
        PyBytes::new_with(py, values.len() * 4, |output| {
            for (bytes, value) in output.as_chunks_mut::<4>().0.iter_mut().zip(values) {
                bytes.copy_from_slice(&value.to_le_bytes());
            }
            Ok(())
        })
    }

    fn save(&mut self, path: &str) -> PyResult<()> {
        self.inner
            .save(Path::new(path))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn load(&mut self, path: &str) -> PyResult<()> {
        self.inner
            .load(Path::new(path))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
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

fn floats(bytes: &[u8]) -> PyResult<Vec<f32>> {
    let (values, remainder) = bytes.as_chunks::<4>();
    if !remainder.is_empty() {
        return Err(PyValueError::new_err("expected little-endian F32 bytes"));
    }
    Ok(values.iter().map(|&v| f32::from_le_bytes(v)).collect())
}

#[pymethods]
impl PyRegressionProbe {
    #[new]
    #[pyo3(signature = (inputs, targets, *, hidden = 128, batch = 64, seed = 0))]
    fn new(
        inputs: usize,
        targets: usize,
        hidden: usize,
        batch: usize,
        seed: u64,
    ) -> PyResult<Self> {
        let inner = RegressionProbe::new(batch, inputs, hidden, targets, seed)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn predict(&mut self, features: &[u8]) -> PyResult<Vec<f32>> {
        self.inner
            .predict(&floats(features)?)
            .map_err(PyValueError::new_err)
    }

    fn reset(&mut self, inputs: usize, targets: usize, seed: u64) -> PyResult<()> {
        self.inner
            .reset(inputs, targets, seed)
            .map_err(PyValueError::new_err)
    }

    fn learn(
        &mut self,
        features: &[u8],
        targets: &[u8],
        mask: &[u8],
        learning_rate: f32,
        regularization: f32,
    ) -> PyResult<f32> {
        self.inner
            .learn(
                &floats(features)?,
                &floats(targets)?,
                &floats(mask)?,
                learning_rate,
                regularization,
            )
            .map_err(PyValueError::new_err)
    }

    fn parameters(&self) -> Vec<Vec<f32>> {
        self.inner.parameters()
    }

    fn set_parameters(&mut self, parameters: Vec<Vec<f32>>) -> PyResult<()> {
        self.inner
            .set_parameters(&parameters)
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
}
