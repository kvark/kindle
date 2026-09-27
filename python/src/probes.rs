use super::*;
use kindle::vision::probe::RegressionProbe;

#[pyclass(name = "RegressionProbe", module = "kindle._native", unsendable)]
pub(crate) struct PyRegressionProbe {
    inner: RegressionProbe,
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
