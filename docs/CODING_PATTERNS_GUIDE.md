# Coding Patterns Guide
## Design Patterns, Traits, Generics, and Best Practices in Rust HFT Arbitrage Lab

This guide documents all advanced programming patterns, design principles, and Rust idioms used throughout the codebase.

---

## Table of Contents
1. [Trait-Based Architecture](#trait-based-architecture)
2. [Generic Programming](#generic-programming)
3. [Type Safety Patterns](#type-safety-patterns)
4. [Memory Management](#memory-management)
5. [Concurrency Patterns](#concurrency-patterns)
6. [Error Handling](#error-handling)
7. [Performance Optimization](#performance-optimization)
8. [FFI & Python Bindings](#ffi--python-bindings)

---

## 1. Trait-Based Architecture

### Strategy Pattern with Traits

**Location:** `rust_python_bindings/src/analytics_bindings/analytics/traits.rs`

```rust
/// Core analyzer trait - Strategy pattern for different algorithms
pub trait Analyzer {
    type Input;    // Associated type for input
    type Output;   // Associated type for output
    
    fn analyze(&self, input: Self::Input) -> AnalyticsResult<Self::Output>;
    fn validate_input(&self, input: &Self::Input) -> AnalyticsResult<()>;
}
```

**Explanation:**
- **Associated Types**: `Input` and `Output` allow each implementation to specify its own types
- **Strategy Pattern**: Different analyzers can be swapped without changing client code
- **Type Safety**: Compile-time guarantees that inputs match expected types

**Example Usage:**
```rust
struct MovingAverageAnalyzer;

impl Analyzer for MovingAverageAnalyzer {
    type Input = Vec<f64>;
    type Output = f64;
    
    fn analyze(&self, input: Self::Input) -> AnalyticsResult<Self::Output> {
        Ok(input.iter().sum::<f64>() / input.len() as f64)
    }
    
    fn validate_input(&self, input: &Self::Input) -> AnalyticsResult<()> {
        if input.is_empty() {
            Err("Empty input".into())
        } else {
            Ok(())
        }
    }
}
```

---

### Matrix Operations Trait

```rust
pub trait MatrixOperation {
    fn compute(&self, data: &Array2<f64>) -> AnalyticsResult<Array2<f64>>;
    fn name(&self) -> &str;
}
```

**Explanation:**
- Abstracts matrix computations (correlation, covariance, etc.)
- Enables polymorphism for different matrix algorithms
- Clean separation between interface and implementation

---

### Rolling Window Calculations

```rust
pub trait RollingWindow {
    fn compute_rolling(&self, data: &Array1<f64>, window: usize) -> AnalyticsResult<Array1<f64>>;
    
    fn min_window_size(&self) -> usize { 2 }  // Default implementation
}
```

**Explanation:**
- **Default Implementations**: Traits can provide default behavior
- Reduces boilerplate in implementors
- Can be overridden when needed

---

## 2. Generic Programming

### Generic Objective Functions (MCMC/MLE)

**Location:** `rust_core/src/mcmc.rs`

```rust
pub fn metropolis_hastings<F>(
    objective: F,
    bounds: &ParameterBounds,
    initial_params: &[f64],
    config: &MCMCConfig,
) -> MCMCResult
where
    F: Fn(&[f64]) -> f64 + Sync,  // Trait bounds
{
    // Implementation uses F without knowing its concrete type
}
```

**Explanation:**
- **Generic Function Type `F`**: Accepts any function matching signature
- **Trait Bounds**: 
  - `Fn(&[f64]) -> f64`: Can be called with slice, returns f64
  - `Sync`: Safe to use across threads (required for parallel execution)
- **Zero-Cost Abstraction**: Compiler monomorphizes at compile time (no runtime overhead)

**Why This Matters:**
```rust
// Can use closures
let result = metropolis_hastings(
    |params| sharpe_ratio(params),
    &bounds,
    &initial,
    &config
);

// Can use function pointers
fn my_objective(params: &[f64]) -> f64 { /* ... */ }
let result = metropolis_hastings(my_objective, &bounds, &initial, &config);

// Can use callable structs
struct MyObjective { data: Vec<f64> }
impl MyObjective {
    fn call(&self, params: &[f64]) -> f64 { /* ... */ }
}
let obj = MyObjective { data: vec![] };
let result = metropolis_hastings(|p| obj.call(p), &bounds, &initial, &config);
```

---

### Generic Array Operations

```rust
use ndarray::{Array1, Array2};

pub trait ArrayProcessor<T> {
    fn process(&self, data: &Array2<T>) -> Result<Array2<T>, Error>;
}
```

**Explanation:**
- Generic over element type `T`
- Works with any numeric type implementing required traits
- Enables code reuse across f32, f64, etc.

---

## 3. Type Safety Patterns

### Newtype Pattern

**Location:** `rust_core/src/orderbook.rs`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Price(pub u64);  // Price in fixed-point (e.g., cents)

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quantity(pub f64);
```

**Explanation:**
- **Wraps primitive types** to add semantic meaning
- **Type Safety**: Cannot accidentally pass Quantity where Price expected
- **Zero Overhead**: Compiler optimizes away the wrapper

**Example:**
```rust
// Compile error - type mismatch!
let price = Price(10000);
let qty = Quantity(1.5);
book.add_order(qty, price);  // ERROR: arguments swapped

// Correct
book.add_order(price, qty);  // OK
```

---

### Phantom Types for Compile-Time State

```rust
use std::marker::PhantomData;

struct Validated;
struct Unvalidated;

struct Parameters<State> {
    values: Vec<f64>,
    _state: PhantomData<State>,
}

impl Parameters<Unvalidated> {
    fn validate(self) -> Result<Parameters<Validated>, Error> {
        // Validation logic
        Ok(Parameters {
            values: self.values,
            _state: PhantomData,
        })
    }
}

impl Parameters<Validated> {
    fn optimize(&self) -> OptimizationResult {
        // Only validated parameters can be optimized
    }
}
```

**Explanation:**
- **PhantomData** marks type without storing data
- Enables state machines at **compile time**
- Cannot call `optimize()` on unvalidated parameters

---

## 4. Memory Management

### Zero-Copy Patterns

**Location:** Python bindings throughout

```rust
use numpy::PyArray1;

#[pyfunction]
fn compute_entropy(py: Python, data: Vec<f64>) -> f64 {
    // Takes ownership of Vec - zero copy from Python
    py.allow_threads(|| {
        entropy(&data, 10)  // Rust owns the data
    })
}
```

**Explanation:**
- PyO3 converts NumPy arrays to `Vec<f64>` without copying when possible
- `py.allow_threads()` releases GIL for true parallelism
- Data is moved, not copied, for maximum efficiency

---

### Arena Allocation Pattern

```rust
struct Arena<T> {
    chunks: Vec<Vec<T>>,
    current_chunk: Vec<T>,
    capacity: usize,
}

impl<T> Arena<T> {
    fn alloc(&mut self, value: T) -> &T {
        if self.current_chunk.len() == self.capacity {
            let new_chunk = Vec::with_capacity(self.capacity);
            self.chunks.push(std::mem::replace(&mut self.current_chunk, new_chunk));
        }
        self.current_chunk.push(value);
        self.current_chunk.last().unwrap()
    }
}
```

**Explanation:**
- **Bulk allocation** reduces malloc overhead
- **Cache-friendly**: Data stored contiguously
- Useful for LOB (Limit Order Book) with many orders

---

### Borrow Splitting

```rust
impl OrderBook {
    pub fn match_order(&mut self, side: Side, qty: f64) -> Vec<Fill> {
        match side {
            Side::Buy => {
                // Borrow only asks side
                let it: Vec<Price> = self.asks.price_iter().collect();
                for p in it {
                    let (f, c, parts) = self.asks.consume_at_price(p, qty);
                    // Process fills
                }
            }
            Side::Sell => {
                // Borrow only bids side
                let it: Vec<Price> = self.bids.price_iter().collect();
                // ...
            }
        }
    }
}
```

**Explanation:**
- **Borrow Checker** prevents simultaneous mutable borrows of entire struct
- **Solution**: Only borrow the field you need (`asks` or `bids`)
- Enables safe concurrent access to different parts of data structure

---

## 5. Concurrency Patterns

### Data Parallelism with Rayon

**Location:** All optimization modules

```rust
use rayon::prelude::*;

// Sequential
let results: Vec<f64> = population.iter()
    .map(|ind| objective(ind))
    .collect();

// Parallel - just add par_iter()!
let results: Vec<f64> = population.par_iter()
    .map(|ind| objective(ind))
    .collect();
```

**Explanation:**
- **Rayon** provides data parallelism with minimal code changes
- Automatically distributes work across CPU cores
- Work-stealing scheduler for load balancing

---

### Message Passing with Channels

**Location:** WebSocket connectors

```rust
use tokio::sync::mpsc;

pub async fn run_binance_ws(tx: Sender<MarketTick>) -> Result<()> {
    let (ws_stream, _) = connect_async(url).await?;
    
    while let Some(msg) = ws_stream.next().await {
        let tick = parse_message(msg?)?;
        tx.send(tick).await?;  // Send to channel
    }
    Ok(())
}
```

**Explanation:**
- **MPSC** (Multi-Producer, Single-Consumer) channels
- **Async**: Non-blocking send/receive
- Decouples producers (WebSocket) from consumers (aggregator)

---

### Thread-Safe Shared State

```rust
use std::sync::{Arc, Mutex};

struct SharedState {
    data: Arc<Mutex<HashMap<String, f64>>>,
}

impl SharedState {
    fn update(&self, key: String, value: f64) {
        let mut data = self.data.lock().unwrap();
        data.insert(key, value);
        // Lock automatically released when `data` goes out of scope
    }
}
```

**Explanation:**
- **Arc** (Atomically Reference Counted): Thread-safe shared ownership
- **Mutex**: Ensures exclusive access
- **RAII**: Lock automatically released

---

## 6. Error Handling

### Result Type and ? Operator

```rust
use std::error::Error;

fn parse_price(s: &str) -> Result<f64, Box<dyn Error>> {
    let value = s.parse::<f64>()?;  // ? propagates error if parse fails
    if value < 0.0 {
        return Err("Negative price".into());
    }
    Ok(value)
}
```

**Explanation:**
- **Result<T, E>**: Explicit error handling
- **? Operator**: Early return on error, automatic type conversion
- **Box<dyn Error>**: Trait object for any error type

---

### Custom Error Types

```rust
#[derive(Debug)]
pub enum OptimizationError {
    InvalidBounds { lower: f64, upper: f64 },
    ConvergenceFailure { iterations: usize },
    NumericalInstability,
}

impl std::fmt::Display for OptimizationError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::InvalidBounds { lower, upper } =>
                write!(f, "Invalid bounds: [{}, {}]", lower, upper),
            // ...
        }
    }
}

impl std::error::Error for OptimizationError {}
```

**Explanation:**
- **Enum** for different error variants
- **Structured errors** carry additional context
- Implements `Display` and `Error` traits for standard error handling

---

### PyO3 Error Handling

```rust
use pyo3::exceptions::PyValueError;

#[pyfunction]
fn fit_hmm(returns: Vec<f64>, n_states: usize) -> PyResult<PyObject> {
    if n_states < 2 {
        return Err(PyValueError::new_err("n_states must be >= 2"));
    }
    // ...
}
```

**Explanation:**
- **PyResult**: Rust's Result type for Python
- **PyValueError**: Maps to Python's ValueError
- Automatic error conversion between Rust and Python

---

## 7. Performance Optimization

### Inline Hints

```rust
#[inline]
fn fast_path(&self, x: f64) -> f64 {
    x * 2.0  // Likely to be inlined by compiler
}

#[inline(always)]
fn critical_path(&self, x: f64) -> f64 {
    x.powi(2)  // Force inline even in debug builds
}

#[inline(never)]
fn large_function(&self) {
    // Prevent inlining of large function
}
```

---

### SIMD (Single Instruction, Multiple Data)

```rust
// Manual SIMD (future enhancement)
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
unsafe fn sum_simd(data: &[f32]) -> f32 {
    let mut sum = _mm_setzero_ps();
    for chunk in data.chunks_exact(4) {
        let v = _mm_loadu_ps(chunk.as_ptr());
        sum = _mm_add_ps(sum, v);
    }
    // Horizontal sum
    let mut result = [0.0f32; 4];
    _mm_storeu_ps(result.as_mut_ptr(), sum);
    result.iter().sum()
}
```

**Explanation:**
- Process multiple data elements in parallel
- 4x-8x speedup for array operations
- Requires unsafe code and platform-specific intrinsics

---

### Cache-Aware Data Structures

```rust
#[repr(align(64))]  // Cache line alignment
struct CacheAlignedData {
    hot_data: [f64; 8],  // Frequently accessed
    cold_data: Vec<f64>, // Rarely accessed
}
```

**Explanation:**
- **Alignment**: Prevents false sharing between threads
- **Hot/Cold Separation**: Improves cache hit rate
- Critical for high-frequency trading

---

### Pre-allocation and Capacity Hints

```rust
fn compute_signature(path: &[Vec<f64>]) -> Vec<f64> {
    let expected_size = estimate_signature_size(path.len());
    let mut sig = Vec::with_capacity(expected_size);  // Pre-allocate
    
    // No reallocation during computation
    for level in 0..max_level {
        sig.extend(compute_level(path, level));
    }
    
    sig
}
```

**Explanation:**
- **with_capacity()**: Allocates memory upfront
- Avoids expensive reallocations during growth
- Significant speedup for large vectors

---

## 8. FFI & Python Bindings

### PyO3 Trait Implementation

```rust
#[pyclass]
struct HMM {
    #[pyo3(get, set)]
    n_states: usize,
    transition_matrix: Vec<Vec<f64>>,
}

#[pymethods]
impl HMM {
    #[new]
    fn new(n_states: usize) -> Self {
        HMM::new_rust(n_states)
    }
    
    fn train(&mut self, data: Vec<f64>, n_iterations: usize) {
        self.train_rust(&data, n_iterations);
    }
}
```

**Explanation:**
- **#[pyclass]**: Makes struct accessible from Python
- **#[pymethods]**: Exports methods to Python
- **#[new]**: Python constructor
- **#[pyo3(get, set)]**: Property access from Python

---

### GIL Release for Performance

```rust
#[pyfunction]
fn expensive_computation(py: Python, data: Vec<f64>) -> f64 {
    py.allow_threads(|| {
        // Rust code runs WITHOUT GIL
        // Python interpreter can run other threads
        compute_intensive_task(&data)
    })
}
```

**Explanation:**
- **GIL** (Global Interpreter Lock) prevents parallel Python execution
- **allow_threads()**: Releases GIL for Rust code
- Enables true multi-threaded parallelism

---

### Python Callback from Rust

```rust
#[pyfunction]
fn optimize(py: Python, objective_fn: PyObject) -> Vec<f64> {
    let objective = |params: &[f64]| -> f64 {
        Python::with_gil(|py| {
            // Re-acquire GIL for Python callback
            objective_fn
                .call1(py, (params.to_vec(),))
                .and_then(|result| result.extract::<f64>(py))
                .unwrap_or(f64::NEG_INFINITY)
        })
    };
    
    differential_evolution(objective, &bounds)
}
```

**Explanation:**
- Rust calls Python function during optimization
- **with_gil()**: Temporarily acquire GIL for callback
- Handles errors gracefully (returns NEG_INFINITY on failure)

---

## Advanced Patterns Summary

### When to Use Each Pattern

| Pattern | Use Case | Complexity |
|---------|----------|------------|
| **Traits** | Polymorphism, abstraction | Medium |
| **Generics** | Code reuse across types | Medium |
| **Newtype** | Type safety, domain modeling | Low |
| **PhantomData** | Compile-time state machines | High |
| **Arena** | Bulk allocations | High |
| **Rayon** | Data parallelism | Low |
| **Channels** | Producer-consumer | Medium |
| **Arc<Mutex>** | Shared state | Medium |
| **Result<?> | Error handling | Low |
| **PyO3 Bindings** | Python interop | Medium |

---

## Best Practices Applied

### 1. **Ownership and Borrowing**
```rust
// ✅ Good: Clear ownership
fn process_data(data: Vec<f64>) -> Vec<f64> {
    data.into_iter().map(|x| x * 2.0).collect()
}

// ✅ Good: Borrow when you don't need ownership
fn sum_data(data: &[f64]) -> f64 {
    data.iter().sum()
}

// ❌ Bad: Unnecessary clone
fn process_data_bad(data: &Vec<f64>) -> Vec<f64> {
    data.clone().into_iter().map(|x| x * 2.0).collect()
}
```

---

### 2. **Iterator Chains**
```rust
// ✅ Good: Lazy, efficient
let result = data.iter()
    .filter(|&&x| x > 0.0)
    .map(|&x| x.sqrt())
    .sum::<f64>();

// ❌ Bad: Intermediate allocations
let positive: Vec<f64> = data.iter().filter(|&&x| x > 0.0).copied().collect();
let sqrt: Vec<f64> = positive.iter().map(|&x| x.sqrt()).collect();
let result: f64 = sqrt.iter().sum();
```

---

### 3. **Error Propagation**
```rust
// ✅ Good: Use ? operator
fn parse_config(path: &str) -> Result<Config, Error> {
    let content = std::fs::read_to_string(path)?;
    let config = serde_json::from_str(&content)?;
    Ok(config)
}

// ❌ Bad: Unwrap can panic
fn parse_config_bad(path: &str) -> Config {
    let content = std::fs::read_to_string(path).unwrap();
    serde_json::from_str(&content).unwrap()
}
```

---

### 4. **Const Generics (Rust 1.51+)**
```rust
struct FixedMatrix<T, const N: usize> {
    data: [[T; N]; N],
}

impl<T: Default + Copy, const N: usize> FixedMatrix<T, N> {
    fn new() -> Self {
        Self {
            data: [[T::default(); N]; N],
        }
    }
}
```

---

## Performance Benchmarking

### Criterion.rs Integration
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_hmm(c: &mut Criterion) {
    let returns = generate_returns(1000);
    
    c.bench_function("hmm_training", |b| {
        b.iter(|| {
            let mut hmm = HMM::new(3);
            hmm.train(black_box(&returns), 100);
        });
    });
}

criterion_group!(benches, benchmark_hmm);
criterion_main!(benches);
```

---

## Continuous Improvement

### Code Review Checklist
- [ ] No unnecessary `mut` declarations
- [ ] No unused imports
- [ ] Error handling with `Result` (no panics in library code)
- [ ] Public API documented with `///` comments
- [ ] Tests cover edge cases
- [ ] Benchmarks for performance-critical code
- [ ] `#[inline]` used judiciously
- [ ] GIL released for compute-intensive Python bindings

---

## Resources

- **Rust Book**: https://doc.rust-lang.org/book/
- **Rust By Example**: https://doc.rust-lang.org/rust-by-example/
- **Rayon**: https://github.com/rayon-rs/rayon
- **PyO3**: https://pyo3.rs/
- **Criterion.rs**: https://github.com/bheisler/criterion.rs

---

## Conclusion

This codebase demonstrates production-grade Rust patterns:
- **Type safety** through newtypes and phantom types
- **Zero-cost abstractions** via traits and generics
- **Performance** through parallelism, SIMD, and cache awareness
- **Correctness** through ownership system and borrow checker
- **Interoperability** with Python via PyO3

All patterns prioritize **performance**, **safety**, and **maintainability** - the core principles of systems programming in Rust.
