//! A borrowed-array boundary; learned operations and updates remain native.

use std::path::Path;

use kindle::vision::levjepa::pretrain::{Batch, Trainer, TrainingConfig};
use numpy::{PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyType,
};

#[pyclass(name = "LeVJepaTrainer", module = "kindle._native", unsendable)]
pub(crate) struct PyLeVJepaTrainer {
    inner: Trainer,
}

fn runtime_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

#[pymethods]
impl PyLeVJepaTrainer {
    #[new]
    fn new(config_json: &str) -> PyResult<Self> {
        let config: TrainingConfig =
            serde_json::from_str(config_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        config.validate().map_err(PyValueError::new_err)?;
        Ok(Self {
            inner: Trainer::new(config, None).map_err(runtime_error)?,
        })
    }

    #[classmethod]
    fn restore(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        Ok(Self {
            inner: Trainer::restore(Path::new(path), None).map_err(runtime_error)?,
        })
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        global_patches: PyReadonlyArray3<'py, f32>,
        global_ids: PyReadonlyArray2<'py, u32>,
        local_patches: PyReadonlyArray3<'py, f32>,
        local_ids: PyReadonlyArray2<'py, u32>,
        directions: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let config = self.inner.config().model;
        for (patches, ids, view) in [
            (&global_patches, &global_ids, config.global()),
            (&local_patches, &local_ids, config.local()),
        ] {
            if patches.shape() != [view.kept, view.clips, 768]
                || ids.shape() != [view.kept, view.clips]
                || !patches.is_c_contiguous()
                || !ids.is_c_contiguous()
            {
                return Err(PyValueError::new_err(
                    "patches/IDs must be C-contiguous token-major arrays",
                ));
            }
        }
        if directions.shape() != [config.projector_output, config.directions]
            || !directions.is_c_contiguous()
        {
            return Err(PyValueError::new_err(
                "directions must be C-contiguous [projector_output, directions]",
            ));
        }
        let batch = Batch {
            global_patches: global_patches.as_slice().map_err(runtime_error)?,
            global_ids: global_ids.as_slice().map_err(runtime_error)?,
            local_patches: local_patches.as_slice().map_err(runtime_error)?,
            local_ids: local_ids.as_slice().map_err(runtime_error)?,
            directions: directions.as_slice().map_err(runtime_error)?,
        };
        let metrics = self.inner.step(batch).map_err(runtime_error)?;
        super::json_to_python(py, &metrics)
    }

    fn save(&mut self, path: &str) -> PyResult<()> {
        self.inner.save(Path::new(path)).map_err(runtime_error)
    }

    fn export_encoder(&mut self, path: &str) -> PyResult<()> {
        self.inner
            .export_encoder(Path::new(path))
            .map_err(runtime_error)
    }

    #[getter]
    fn completed_steps(&self) -> u32 {
        self.inner.completed_steps()
    }

    #[getter]
    fn config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        super::json_to_python(py, &self.inner.config())
    }

    #[getter]
    fn gpu_device<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        super::json_to_python(py, &self.inner.device_info())
    }
}
