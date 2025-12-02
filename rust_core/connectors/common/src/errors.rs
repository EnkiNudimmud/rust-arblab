use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum ConnectorError {
    #[error("network error: {0}")]
    Network(String),

    #[error("parse error: {0}")]
    Parse(String),

    #[error("rate limited: {0}")]
    RateLimited(String),

    #[error("server error ({0}): {1}")]
    ServerError(u16, String),

    #[error("client error ({0}): {1}")]
    ClientError(u16, String),

    #[error("timeout: {0}")]
    Timeout(String),

    #[error("other: {0}")]
    Other(String),
}

impl ConnectorError {
    /// Returns true if the error is retriable (transient network issues, rate limits, server errors)
    pub fn is_retriable(&self) -> bool {
        match self {
            ConnectorError::Network(_) => true,
            ConnectorError::RateLimited(_) => true,
            ConnectorError::ServerError(code, _) => *code >= 500,
            ConnectorError::Timeout(_) => true,
            ConnectorError::Parse(_) => false,
            ConnectorError::ClientError(code, _) => {
                // 408 Request Timeout and 429 Too Many Requests are retriable
                *code == 408 || *code == 429
            }
            ConnectorError::Other(_) => false,
        }
    }

    /// Create a ConnectorError from an HTTP status code and message
    pub fn from_http_status(status_code: u16, message: String) -> Self {
        match status_code {
            429 => ConnectorError::RateLimited(message),
            400..=499 => ConnectorError::ClientError(status_code, message),
            500..=599 => ConnectorError::ServerError(status_code, message),
            _ => ConnectorError::Other(format!("HTTP {}: {}", status_code, message)),
        }
    }
}
