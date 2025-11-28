use connectors_common::errors::ConnectorError;
use connectors_common::retry::{with_retry, RetryConfig};
use connectors_common::types::{OrderBookLevel, OrderBookSnapshot};
use reqwest::Client;
use std::time::Duration;

/// Internal function to perform a single orderbook fetch attempt
async fn fetch_orderbook_once(client: &Client, url: &str, pair: &str) -> Result<OrderBookSnapshot, ConnectorError> {
    let response = client
        .get(url)
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .map_err(|e| {
            if e.is_timeout() {
                ConnectorError::Timeout(format!("Request to {} timed out", url))
            } else if e.is_connect() {
                ConnectorError::Network(format!("Connection error: {}", e))
            } else {
                ConnectorError::Network(format!("Request error: {}", e))
            }
        })?;

    let status = response.status();
    if !status.is_success() {
        let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
        return Err(ConnectorError::from_http_status(status.as_u16(), error_text));
    }

    let text = response
        .text()
        .await
        .map_err(|e| ConnectorError::Network(format!("Failed to read response body: {}", e)))?;

    let v: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| ConnectorError::Parse(format!("Invalid JSON: {}", e)))?;

    let mut bids = vec![];
    let mut asks = vec![];

    if let Some(result) = v.get("result") {
        if let Some((_, book)) = result.as_object().and_then(|m| m.iter().next()) {
            if let Some(b) = book.get("bids").and_then(|v| v.as_array()) {
                for it in b.iter().take(5) {
                    if let (Some(p), Some(q)) = (
                        it.get(0).and_then(|s| s.as_str()),
                        it.get(1).and_then(|s| s.as_str()),
                    ) {
                        bids.push(OrderBookLevel {
                            price: p.parse().unwrap_or(0.0),
                            qty: q.parse().unwrap_or(0.0),
                        });
                    }
                }
            }
            if let Some(a) = book.get("asks").and_then(|v| v.as_array()) {
                for it in a.iter().take(5) {
                    if let (Some(p), Some(q)) = (
                        it.get(0).and_then(|s| s.as_str()),
                        it.get(1).and_then(|s| s.as_str()),
                    ) {
                        asks.push(OrderBookLevel {
                            price: p.parse().unwrap_or(0.0),
                            qty: q.parse().unwrap_or(0.0),
                        });
                    }
                }
            }
        }
    }

    Ok(OrderBookSnapshot {
        exchange: "kraken".to_string(),
        pair: pair.to_string(),
        bids,
        asks,
        ts: chrono::Utc::now().timestamp_millis() as u128,
    })
}

/// Fetch orderbook from Kraken with automatic retry on transient failures.
///
/// This function implements exponential backoff retry logic for handling:
/// - Network connectivity issues
/// - Rate limiting (HTTP 429)
/// - Server errors (HTTP 5xx)
/// - Request timeouts
///
/// # Arguments
///
/// * `pair` - Trading pair symbol (e.g., "XBTUSD")
///
/// # Returns
///
/// OrderBookSnapshot on success, or an error after all retries are exhausted.
pub async fn fetch_orderbook(pair: &str) -> Result<OrderBookSnapshot, ConnectorError> {
    let url = format!(
        "https://api.kraken.com/0/public/Depth?pair={}&count=5",
        pair
    );
    let client = Client::new();
    let config = RetryConfig::default();

    with_retry(config, || {
        let client = &client;
        let url = &url;
        async move { fetch_orderbook_once(client, url, pair).await }
    })
    .await
}

/// Fetch orderbook with custom retry configuration.
///
/// Use this when you need more control over retry behavior.
pub async fn fetch_orderbook_with_config(
    pair: &str,
    config: RetryConfig,
) -> Result<OrderBookSnapshot, ConnectorError> {
    let url = format!(
        "https://api.kraken.com/0/public/Depth?pair={}&count=5",
        pair
    );
    let client = Client::new();

    with_retry(config, || {
        let client = &client;
        let url = &url;
        async move { fetch_orderbook_once(client, url, pair).await }
    })
    .await
}
