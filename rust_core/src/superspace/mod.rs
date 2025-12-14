// Superspace Anomaly Detection Module
// Implements supersymmetry, ghost fields, and topological invariants for market analysis

pub mod grassmann;
pub mod ghost_fields;
pub mod chern_simons;
pub mod superspace_14d;
pub mod anomaly_detector;

pub use grassmann::GrassmannNumber;
pub use ghost_fields::{GhostFieldSystem, GhostFieldParams};
pub use chern_simons::ChernSimonsCalculator;
pub use superspace_14d::Superspace14D;
pub use anomaly_detector::AnomalyDetector;
