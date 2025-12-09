//! Retry utilities with exponential backoff for HTTP requests.
//!
//! This module provides utilities to retry failed requests with configurable
//! exponential backoff, which is essential for handling transient network errors
//! and rate limiting in API calls.

use std::future::Future;
use std::time::Duration;

use crate::errors::ConnectorError;

/// Configuration for retry behavior with exponential backoff.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts (excluding the initial attempt)
    pub max_retries: u32,
    /// Initial delay before first retry
    pub initial_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Multiplier for exponential backoff (e.g., 2.0 doubles delay each retry)
    pub backoff_multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 2.0,
        }
    }
}

impl RetryConfig {
    /// Create a new retry configuration
    pub fn new(max_retries: u32, initial_delay_ms: u64, max_delay_secs: u64) -> Self {
        Self {
            max_retries,
            initial_delay: Duration::from_millis(initial_delay_ms),
            max_delay: Duration::from_secs(max_delay_secs),
            backoff_multiplier: 2.0,
        }
    }

    /// Create a configuration for aggressive retries (good for HFT scenarios)
    pub fn aggressive() -> Self {
        Self {
            max_retries: 5,
            initial_delay: Duration::from_millis(50),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 1.5,
        }
    }

    /// Create a configuration for conservative retries (good for rate-limited APIs)
    pub fn conservative() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
        }
    }

    /// Calculate the delay for a given attempt number (0-indexed)
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        if attempt == 0 {
            return Duration::ZERO;
        }

        let delay_ms = self.initial_delay.as_millis() as f64
            * self.backoff_multiplier.powi((attempt - 1) as i32);
        let delay = Duration::from_millis(delay_ms as u64);

        std::cmp::min(delay, self.max_delay)
    }
}

/// Execute a future with retry logic using exponential backoff.
///
/// This function will retry the operation if it returns a retriable error
/// (as determined by `ConnectorError::is_retriable()`).
///
/// # Arguments
///
/// * `config` - Retry configuration
/// * `operation` - A closure that returns a future producing `Result<T, ConnectorError>`
///
/// # Returns
///
/// The result of the operation, or the last error if all retries failed.
///
/// # Example
///
/// ```ignore
/// use connectors_common::retry::{RetryConfig, with_retry};
///
/// let result = with_retry(RetryConfig::default(), || async {
///     fetch_orderbook("BTCUSDT").await
/// }).await;
/// ```
pub async fn with_retry<T, F, Fut>(config: RetryConfig, mut operation: F) -> Result<T, ConnectorError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, ConnectorError>>,
{
    let mut last_error: Option<ConnectorError> = None;

    for attempt in 0..=config.max_retries {
        // Wait before retry (skip delay on first attempt)
        let delay = config.delay_for_attempt(attempt);
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }

        match operation().await {
            Ok(result) => return Ok(result),
            Err(e) => {
                if e.is_retriable() && attempt < config.max_retries {
                    // Log retry attempt (if logging is available)
                    last_error = Some(e);
                    continue;
                }
                // Non-retriable error or final attempt
                return Err(e);
            }
        }
    }

    // This should not be reachable, but handle it gracefully
    Err(last_error.unwrap_or_else(|| ConnectorError::Other("Retry logic error".to_string())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_retry_config_delay_calculation() {
        let config = RetryConfig {
            max_retries: 5,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 2.0,
        };

        assert_eq!(config.delay_for_attempt(0), Duration::ZERO);
        assert_eq!(config.delay_for_attempt(1), Duration::from_millis(100));
        assert_eq!(config.delay_for_attempt(2), Duration::from_millis(200));
        assert_eq!(config.delay_for_attempt(3), Duration::from_millis(400));
        assert_eq!(config.delay_for_attempt(4), Duration::from_millis(800));
    }

    #[test]
    fn test_retry_config_max_delay_cap() {
        let config = RetryConfig {
            max_retries: 10,
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 2.0,
        };

        // After several attempts, delay should be capped at max_delay
        assert_eq!(config.delay_for_attempt(5), Duration::from_secs(5));
        assert_eq!(config.delay_for_attempt(10), Duration::from_secs(5));
    }

    #[tokio::test]
    async fn test_with_retry_success_first_attempt() {
        let config = RetryConfig::default();
        let attempt_count = Arc::new(AtomicU32::new(0));
        let attempt_count_clone = attempt_count.clone();

        let result: Result<String, ConnectorError> = with_retry(config, || {
            let count = attempt_count_clone.clone();
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                Ok("success".to_string())
            }
        })
        .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
        assert_eq!(attempt_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_with_retry_success_after_retries() {
        let config = RetryConfig {
            max_retries: 3,
            initial_delay: Duration::from_millis(1), // Very short for testing
            max_delay: Duration::from_millis(10),
            backoff_multiplier: 2.0,
        };

        let attempt_count = Arc::new(AtomicU32::new(0));
        let attempt_count_clone = attempt_count.clone();

        let result: Result<String, ConnectorError> = with_retry(config, || {
            let count = attempt_count_clone.clone();
            async move {
                let current = count.fetch_add(1, Ordering::SeqCst);
                if current < 2 {
                    // Fail first two attempts with retriable error
                    Err(ConnectorError::Network("temporary failure".to_string()))
                } else {
                    Ok("success".to_string())
                }
            }
        })
        .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
        assert_eq!(attempt_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_with_retry_non_retriable_error() {
        let config = RetryConfig {
            max_retries: 3,
            initial_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(10),
            backoff_multiplier: 2.0,
        };

        let attempt_count = Arc::new(AtomicU32::new(0));
        let attempt_count_clone = attempt_count.clone();

        let result: Result<String, ConnectorError> = with_retry(config, || {
            let count = attempt_count_clone.clone();
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                // Parse error is not retriable
                Err(ConnectorError::Parse("invalid JSON".to_string()))
            }
        })
        .await;

        assert!(result.is_err());
        // Should fail immediately without retries
        assert_eq!(attempt_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_with_retry_exhausted() {
        let config = RetryConfig {
            max_retries: 2,
            initial_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(10),
            backoff_multiplier: 2.0,
        };

        let attempt_count = Arc::new(AtomicU32::new(0));
        let attempt_count_clone = attempt_count.clone();

        let result: Result<String, ConnectorError> = with_retry(config, || {
            let count = attempt_count_clone.clone();
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                Err(ConnectorError::Network("persistent failure".to_string()))
            }
        })
        .await;

        assert!(result.is_err());
        // Initial attempt + 2 retries = 3 total attempts
        assert_eq!(attempt_count.load(Ordering::SeqCst), 3);
    }
}
