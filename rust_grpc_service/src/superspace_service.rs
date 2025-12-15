use tonic::{Request, Response, Status};
use crate::hft::superspace::superspace_service_server::SuperspaceService;
use crate::hft::superspace::{
    DetectRequest, DetectResponse, CsRequest, CsResponse,
    GhostFieldRequest, GhostFieldResponse,
    Anomaly, Signal, ArrayWrapper,
};
use rust_core::superspace::*;

#[derive(Debug, Default)]
pub struct MySuperspaceService;

#[tonic::async_trait]
impl SuperspaceService for MySuperspaceService {
    async fn detect_anomalies(
        &self,
        request: Request<DetectRequest>,
    ) -> Result<Response<DetectResponse>, Status> {
        let req = request.into_inner();
        let params = req.params.ok_or(Status::invalid_argument("Missing ghost params"))?;
        let det_params = req.detector_params.ok_or(Status::invalid_argument("Missing detector params"))?;
        
        // Reconstruct GhostFieldSystem
        let ghost_system = if !req.bosonic_coords.is_empty() {
            let bosonic: Vec<Vec<f64>> = req.bosonic_coords.iter()
                .map(|bg| bg.values.clone())
                .collect();
                
            let core_params = GhostFieldParams {
                spring_constant: params.spring_constant,
                damping: params.damping,
                noise_level: params.noise_level,
                n_modes: params.n_modes as usize,
            };
            
            GhostFieldSystem::from_bosonic_coords(&bosonic, core_params)
        } else {
            // Can't detect without system or CS values + system
             return Err(Status::invalid_argument("Bosonic coords required to rebuild system state"));
        };
        
        // Use supplied CS values or calculate them
        // For now, assuming anomaly detection needs both system state and CS values
        // which usually come from the system.
        
        let _calc = ChernSimonsCalculator::new(1.0).with_window(30);
        let cs_values = if req.cs_values.is_empty() {
            // Need to calculate CS from system fields
            // Simplified logic here; in real app we'd iterate over time
            vec![] // Placeholder
        } else {
            req.cs_values
        };

        let detector = AnomalyDetector::new(
            det_params.threshold,
            det_params.alpha,
            det_params.beta,
        );
        
        let anomalies = detector.detect(&ghost_system, &cs_values);
        let signals = detector.generate_signals(&anomalies);
        
        let proto_anomalies = anomalies.into_iter()
            .map(|(t, s)| Anomaly { time_index: t as i32, score: s })
            .collect();
            
        let proto_signals = signals.into_iter()
            .map(|(t, s)| Signal { time_index: t as i32, signal_type: s as i32 })
            .collect();

        Ok(Response::new(DetectResponse {
            anomalies: proto_anomalies,
            signals: proto_signals,
        }))
    }

    async fn calculate_chern_simons(
        &self,
        request: Request<CsRequest>,
    ) -> Result<Response<CsResponse>, Status> {
        let req = request.into_inner();
        let calc = ChernSimonsCalculator::new(req.coupling).with_window(req.window as usize);
        let values = calc.calculate(&req.field);
        // Note: transitions detection logic is separate in core
        
        Ok(Response::new(CsResponse {
            values,
            transitions: vec![], // Placeholder
        }))
    }

    async fn generate_ghost_field(
        &self,
        request: Request<GhostFieldRequest>,
    ) -> Result<Response<GhostFieldResponse>, Status> {
        let req = request.into_inner();
        let params = req.params.ok_or(Status::invalid_argument("Missing params"))?;
        
        let bosonic: Vec<Vec<f64>> = req.bosonic_coords.iter()
            .map(|bg| bg.values.clone())
            .collect();
            
        let core_params = GhostFieldParams {
            spring_constant: params.spring_constant,
            damping: params.damping,
            noise_level: params.noise_level,
            n_modes: params.n_modes as usize,
        };
        
        let system = GhostFieldSystem::from_bosonic_coords(&bosonic, core_params);
        
        let fields = system.fields.iter()
            .map(|v| ArrayWrapper { values: v.clone() })
            .collect();
            
        Ok(Response::new(GhostFieldResponse {
            fields,
            divergence: system.divergence,
            curl: system.curl,
        }))
    }
}
