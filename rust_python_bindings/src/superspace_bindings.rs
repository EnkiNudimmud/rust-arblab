// Python bindings for Superspace Anomaly Detection

use pyo3::prelude::*;
use rust_core::superspace::*;

// ============================================================================
// GrassmannNumber Bindings
// ============================================================================

#[pyclass(name = "PyGrassmannNumber")]
#[derive(Clone)]
pub struct PyGrassmannNumber {
    inner: GrassmannNumber,
}

#[pymethods]
impl PyGrassmannNumber {
    #[new]
    fn new(scalar: f64, grassmann: Vec<f64>) -> Self {
        Self {
            inner: GrassmannNumber::new(scalar, grassmann),
        }
    }

    fn norm(&self) -> f64 {
        self.inner.norm()
    }

    fn __add__(&self, other: &PyGrassmannNumber) -> PyGrassmannNumber {
        PyGrassmannNumber {
            inner: self.inner.clone() + other.inner.clone(),
        }
    }

    fn __sub__(&self, other: &PyGrassmannNumber) -> PyGrassmannNumber {
        PyGrassmannNumber {
            inner: self.inner.clone() - other.inner.clone(),
        }
    }

    fn __mul__(&self, other: &PyGrassmannNumber) -> PyGrassmannNumber {
        PyGrassmannNumber {
            inner: self.inner.clone() * other.inner.clone(),
        }
    }

    #[getter]
    fn scalar(&self) -> f64 {
        self.inner.scalar
    }

    #[getter]
    fn grassmann(&self) -> Vec<f64> {
        self.inner.grassmann.clone()
    }

    fn __repr__(&self) -> String {
        format!("GrassmannNumber(scalar={}, grassmann={:?})", 
                self.inner.scalar, self.inner.grassmann)
    }
}

// ============================================================================
// GhostFieldParams Bindings
// ============================================================================

#[pyclass(name = "PyGhostFieldParams")]
#[derive(Clone)]
pub struct PyGhostFieldParams {
    inner: GhostFieldParams,
}

#[pymethods]
impl PyGhostFieldParams {
    #[new]
    #[pyo3(signature = (spring_constant=1.0, damping=0.1, noise_level=0.01, n_modes=7))]
    fn new(
        spring_constant: f64,
        damping: f64,
        noise_level: f64,
        n_modes: usize,
    ) -> Self {
        Self {
            inner: GhostFieldParams {
                spring_constant,
                damping,
                noise_level,
                n_modes,
            },
        }
    }

    #[getter]
    fn spring_constant(&self) -> f64 {
        self.inner.spring_constant
    }

    #[getter]
    fn damping(&self) -> f64 {
        self.inner.damping
    }

    #[getter]
    fn noise_level(&self) -> f64 {
        self.inner.noise_level
    }

    #[getter]
    fn n_modes(&self) -> usize {
        self.inner.n_modes
    }
}

// ============================================================================
// GhostFieldSystem Bindings
// ============================================================================

#[pyclass(name = "PyGhostFieldSystem")]
pub struct PyGhostFieldSystem {
    inner: GhostFieldSystem,
}

#[pymethods]
impl PyGhostFieldSystem {
    #[staticmethod]
    fn from_bosonic_coords(
        bosonic: Vec<Vec<f64>>,
        params: PyGhostFieldParams,
    ) -> Self {
        Self {
            inner: GhostFieldSystem::from_bosonic_coords(&bosonic, params.inner),
        }
    }

    fn get_field(&self, time: usize, mode: usize) -> f64 {
        self.inner.get_field(time, mode)
    }

    fn get_divergence_at(&self, time: usize) -> f64 {
        self.inner.get_divergence(time)
    }

    fn get_curl_at(&self, time: usize) -> f64 {
        self.inner.get_curl(time)
    }

    #[getter]
    fn fields(&self) -> Vec<Vec<f64>> {
        self.inner.fields.clone()
    }

    #[getter]
    fn divergence(&self) -> Vec<f64> {
        self.inner.divergence.clone()
    }

    #[getter]
    fn curl(&self) -> Vec<f64> {
        self.inner.curl.clone()
    }

    fn __repr__(&self) -> String {
        format!("GhostFieldSystem(n_time={}, n_modes={})", 
                self.inner.fields.len(),
                if self.inner.fields.is_empty() { 0 } else { self.inner.fields[0].len() })
    }
}

// ============================================================================
// ChernSimonsCalculator Bindings
// ============================================================================

#[pyclass(name = "PyChernSimonsCalculator")]
pub struct PyChernSimonsCalculator {
    inner: ChernSimonsCalculator,
}

#[pymethods]
impl PyChernSimonsCalculator {
    #[new]
    #[pyo3(signature = (coupling=1.0, window=30))]
    fn new(coupling: f64, window: usize) -> Self {
        Self {
            inner: ChernSimonsCalculator::new(coupling).with_window(window),
        }
    }

    fn calculate(&self, field: Vec<f64>) -> Vec<f64> {
        self.inner.calculate(&field)
    }

    fn detect_transitions(&self, cs_values: Vec<f64>, threshold: f64) -> Vec<usize> {
        self.inner.detect_transitions(&cs_values, threshold)
    }

    #[getter]
    fn coupling(&self) -> f64 {
        self.inner.coupling
    }

    #[getter]
    fn window(&self) -> usize {
        self.inner.window
    }

    fn __repr__(&self) -> String {
        format!("ChernSimonsCalculator(coupling={}, window={})",
                self.inner.coupling, self.inner.window)
    }
}

// ============================================================================
// Superspace14D Bindings
// ============================================================================

#[pyclass(name = "PySuperspace14D")]
pub struct PySuperspace14D {
    inner: Superspace14D,
}

#[pymethods]
impl PySuperspace14D {
    #[new]
    fn new(bosonic: Vec<f64>, fermionic: Vec<PyGrassmannNumber>) -> PyResult<Self> {
        let fermionic_inner: Vec<GrassmannNumber> = fermionic
            .into_iter()
            .map(|g| g.inner)
            .collect();
        
        Ok(Self {
            inner: Superspace14D::new(bosonic, fermionic_inner),
        })
    }

    #[staticmethod]
    fn zero() -> Self {
        Self {
            inner: Superspace14D::zero(),
        }
    }

    fn distance(&self, other: &PySuperspace14D) -> f64 {
        self.inner.distance(&other.inner)
    }

    fn brst_transform(&self) -> PySuperspace14D {
        PySuperspace14D {
            inner: self.inner.brst_transform(),
        }
    }

    fn gauge_transform(&self, alpha: f64) -> PySuperspace14D {
        PySuperspace14D {
            inner: self.inner.gauge_transform(alpha),
        }
    }

    #[getter]
    fn bosonic(&self) -> Vec<f64> {
        self.inner.bosonic.clone()
    }

    fn __repr__(&self) -> String {
        format!("Superspace14D(bosonic={:?})", &self.inner.bosonic[..3])
    }
}

// ============================================================================
// AnomalyDetector Bindings
// ============================================================================

#[pyclass(name = "PyAnomalyDetector")]
pub struct PyAnomalyDetector {
    inner: AnomalyDetector,
}

#[pymethods]
impl PyAnomalyDetector {
    #[new]
    #[pyo3(signature = (threshold=2.5, alpha=0.5, beta=0.5))]
    fn new(threshold: f64, alpha: f64, beta: f64) -> Self {
        Self {
            inner: AnomalyDetector::new(threshold, alpha, beta),
        }
    }

    fn detect(
        &self,
        ghost_system: &PyGhostFieldSystem,
        cs_values: Vec<f64>,
    ) -> Vec<(usize, f64)> {
        self.inner.detect(&ghost_system.inner, &cs_values)
    }

    fn generate_signals(&self, anomalies: Vec<(usize, f64)>) -> Vec<(usize, i8)> {
        self.inner.generate_signals(&anomalies)
    }

    #[getter]
    fn threshold(&self) -> f64 {
        self.inner.threshold
    }

    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.alpha
    }

    #[getter]
    fn beta(&self) -> f64 {
        self.inner.beta
    }

    fn __repr__(&self) -> String {
        format!("AnomalyDetector(threshold={}, alpha={}, beta={})",
                self.inner.threshold, self.inner.alpha, self.inner.beta)
    }
}

// ============================================================================
// Module Registration
// ============================================================================

pub fn register_superspace_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let superspace = PyModule::new_bound(parent_module.py(), "superspace")?;
    
    superspace.add_class::<PyGrassmannNumber>()?;
    superspace.add_class::<PyGhostFieldParams>()?;
    superspace.add_class::<PyGhostFieldSystem>()?;
    superspace.add_class::<PyChernSimonsCalculator>()?;
    superspace.add_class::<PySuperspace14D>()?;
    superspace.add_class::<PyAnomalyDetector>()?;
    
    parent_module.add_submodule(&superspace)?;
    
    Ok(())
}
