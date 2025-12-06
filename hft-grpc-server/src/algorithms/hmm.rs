/// Hidden Markov Model Implementation
/// 
/// Baum-Welch training and Viterbi decoding for regime detection

use std::f64;

/// Fit HMM using Baum-Welch algorithm
pub fn fit_hmm(
    observations: &[f64],
    n_states: usize,
    n_iterations: usize,
    tolerance: f64,
) -> HMMResult {
    if observations.len() < 3 || n_states == 0 {
        return HMMResult::default();
    }
    
    let n_obs = observations.len();
    
    // Initialize parameters
    let mut params = initialize_hmm(observations, n_states);
    
    let mut prev_log_likelihood = f64::NEG_INFINITY;
    let mut converged = false;
    
    for iter in 0..n_iterations {
        // E-step: Forward-Backward
        let alpha = forward(observations, &params);
        let beta = backward(observations, &params);
        let gamma = compute_gamma(&alpha, &beta);
        let xi = compute_xi(observations, &params, &alpha, &beta);
        
        // M-step: Update parameters
        update_parameters(observations, &mut params, &gamma, &xi);
        
        // Check convergence
        let log_likelihood = compute_log_likelihood(&alpha);
        
        if (log_likelihood - prev_log_likelihood).abs() < tolerance {
            converged = true;
            break;
        }
        
        prev_log_likelihood = log_likelihood;
    }
    
    // Get current state probabilities
    let alpha = forward(observations, &params);
    let last_alpha = &alpha[n_obs - 1];
    let sum: f64 = last_alpha.iter().sum();
    let state_probabilities: Vec<f64> = last_alpha.iter()
        .map(|&p| p / sum)
        .collect();
    
    HMMResult {
        state_probabilities,
        transition_matrix: flatten_matrix(&params.transition_matrix),
        emission_means: params.emission_means.clone(),
        emission_stds: params.emission_stds.clone(),
        log_likelihood: prev_log_likelihood,
        converged,
    }
}

fn initialize_hmm(observations: &[f64], n_states: usize) -> HMMParams {
    let n_obs = observations.len();
    
    // Initialize transition matrix (uniform)
    let transition_matrix = vec![vec![1.0 / n_states as f64; n_states]; n_states];
    
    // Initialize initial probabilities (uniform)
    let initial_probs = vec![1.0 / n_states as f64; n_states];
    
    // Initialize emission parameters using quantiles
    let mut sorted_obs = observations.to_vec();
    sorted_obs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let mut emission_means = Vec::with_capacity(n_states);
    let mut emission_stds = Vec::with_capacity(n_states);
    
    for i in 0..n_states {
        let start_idx = (i * n_obs) / n_states;
        let end_idx = ((i + 1) * n_obs) / n_states;
        let segment = &sorted_obs[start_idx..end_idx];
        
        let mean = segment.iter().sum::<f64>() / segment.len() as f64;
        let var: f64 = segment.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / segment.len() as f64;
        let std = var.sqrt().max(1e-6);
        
        emission_means.push(mean);
        emission_stds.push(std);
    }
    
    HMMParams {
        n_states,
        transition_matrix,
        emission_means,
        emission_stds,
        initial_probs,
    }
}

fn forward(observations: &[f64], params: &HMMParams) -> Vec<Vec<f64>> {
    let n_obs = observations.len();
    let n_states = params.n_states;
    let mut alpha = vec![vec![0.0; n_states]; n_obs];
    
    // t=0
    for s in 0..n_states {
        alpha[0][s] = params.initial_probs[s] * emission_prob(observations[0], params, s);
    }
    
    // Normalize
    let sum0: f64 = alpha[0].iter().sum();
    if sum0 > 0.0 {
        alpha[0].iter_mut().for_each(|p| *p /= sum0);
    }
    
    // t=1..T-1
    for t in 1..n_obs {
        for s in 0..n_states {
            let mut sum = 0.0;
            for prev_s in 0..n_states {
                sum += alpha[t-1][prev_s] * params.transition_matrix[prev_s][s];
            }
            alpha[t][s] = sum * emission_prob(observations[t], params, s);
        }
        
        // Normalize
        let sum_t: f64 = alpha[t].iter().sum();
        if sum_t > 0.0 {
            alpha[t].iter_mut().for_each(|p| *p /= sum_t);
        }
    }
    
    alpha
}

fn backward(observations: &[f64], params: &HMMParams) -> Vec<Vec<f64>> {
    let n_obs = observations.len();
    let n_states = params.n_states;
    let mut beta = vec![vec![0.0; n_states]; n_obs];
    
    // t=T-1
    for s in 0..n_states {
        beta[n_obs - 1][s] = 1.0;
    }
    
    // t=T-2..0
    for t in (0..n_obs-1).rev() {
        for s in 0..n_states {
            let mut sum = 0.0;
            for next_s in 0..n_states {
                sum += params.transition_matrix[s][next_s]
                    * emission_prob(observations[t+1], params, next_s)
                    * beta[t+1][next_s];
            }
            beta[t][s] = sum;
        }
        
        // Normalize
        let sum_t: f64 = beta[t].iter().sum();
        if sum_t > 0.0 {
            beta[t].iter_mut().for_each(|p| *p /= sum_t);
        }
    }
    
    beta
}

fn compute_gamma(alpha: &[Vec<f64>], beta: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n_obs = alpha.len();
    let n_states = alpha[0].len();
    let mut gamma = vec![vec![0.0; n_states]; n_obs];
    
    for t in 0..n_obs {
        let mut sum = 0.0;
        for s in 0..n_states {
            gamma[t][s] = alpha[t][s] * beta[t][s];
            sum += gamma[t][s];
        }
        
        if sum > 0.0 {
            gamma[t].iter_mut().for_each(|p| *p /= sum);
        }
    }
    
    gamma
}

fn compute_xi(
    observations: &[f64],
    params: &HMMParams,
    alpha: &[Vec<f64>],
    beta: &[Vec<f64>],
) -> Vec<Vec<Vec<f64>>> {
    let n_obs = observations.len();
    let n_states = params.n_states;
    let mut xi = vec![vec![vec![0.0; n_states]; n_states]; n_obs - 1];
    
    for t in 0..n_obs-1 {
        let mut sum = 0.0;
        
        for i in 0..n_states {
            for j in 0..n_states {
                xi[t][i][j] = alpha[t][i]
                    * params.transition_matrix[i][j]
                    * emission_prob(observations[t+1], params, j)
                    * beta[t+1][j];
                sum += xi[t][i][j];
            }
        }
        
        if sum > 0.0 {
            for i in 0..n_states {
                for j in 0..n_states {
                    xi[t][i][j] /= sum;
                }
            }
        }
    }
    
    xi
}

fn update_parameters(
    observations: &[f64],
    params: &mut HMMParams,
    gamma: &[Vec<f64>],
    xi: &[Vec<Vec<f64>>],
) {
    let n_obs = observations.len();
    let n_states = params.n_states;
    
    // Update initial probabilities
    for s in 0..n_states {
        params.initial_probs[s] = gamma[0][s];
    }
    
    // Update transition matrix
    for i in 0..n_states {
        let mut sum_gamma = 0.0;
        for t in 0..n_obs-1 {
            sum_gamma += gamma[t][i];
        }
        
        if sum_gamma > 1e-10 {
            for j in 0..n_states {
                let mut sum_xi = 0.0;
                for t in 0..n_obs-1 {
                    sum_xi += xi[t][i][j];
                }
                params.transition_matrix[i][j] = sum_xi / sum_gamma;
            }
        }
    }
    
    // Update emission parameters
    for s in 0..n_states {
        let mut sum_gamma = 0.0;
        let mut sum_obs = 0.0;
        
        for t in 0..n_obs {
            sum_gamma += gamma[t][s];
            sum_obs += gamma[t][s] * observations[t];
        }
        
        if sum_gamma > 1e-10 {
            params.emission_means[s] = sum_obs / sum_gamma;
            
            let mut sum_var = 0.0;
            for t in 0..n_obs {
                let diff = observations[t] - params.emission_means[s];
                sum_var += gamma[t][s] * diff * diff;
            }
            params.emission_stds[s] = (sum_var / sum_gamma).sqrt().max(1e-6);
        }
    }
}

fn emission_prob(obs: f64, params: &HMMParams, state: usize) -> f64 {
    let mean = params.emission_means[state];
    let std = params.emission_stds[state];
    
    // Gaussian PDF
    let z = (obs - mean) / std;
    let coef = 1.0 / (std * (2.0 * std::f64::consts::PI).sqrt());
    coef * (-0.5 * z * z).exp()
}

fn compute_log_likelihood(alpha: &[Vec<f64>]) -> f64 {
    let last_alpha = &alpha[alpha.len() - 1];
    let sum: f64 = last_alpha.iter().sum();
    sum.ln()
}

fn flatten_matrix(matrix: &[Vec<f64>]) -> Vec<f64> {
    matrix.iter().flat_map(|row| row.iter().copied()).collect()
}

#[derive(Debug, Clone)]
struct HMMParams {
    n_states: usize,
    transition_matrix: Vec<Vec<f64>>,
    emission_means: Vec<f64>,
    emission_stds: Vec<f64>,
    initial_probs: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct HMMResult {
    pub state_probabilities: Vec<f64>,
    pub transition_matrix: Vec<f64>,
    pub emission_means: Vec<f64>,
    pub emission_stds: Vec<f64>,
    pub log_likelihood: f64,
    pub converged: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hmm_basic() {
        let observations = vec![0.1, 0.2, 0.15, 2.0, 1.9, 2.1, 0.1, 0.2];
        let result = fit_hmm(&observations, 2, 10, 1e-4);
        assert_eq!(result.emission_means.len(), 2);
        assert_eq!(result.state_probabilities.len(), 2);
    }
}
