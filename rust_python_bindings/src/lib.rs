use pyo3::prelude::*;
#[pyfunction] fn hello()->PyResult<String>{ Ok("hft_py ready".into()) }
#[pymodule] fn hft_py(_py: Python, m: &PyModule)->PyResult<()> { m.add_function(wrap_pyfunction!(hello, m)?)?; Ok(()) }
