use connectors_common::errors::ConnectorError;
use connectors_common::retry::{with_retry, RetryConfig};
use connectors_common::types::MarketTick;
use log::info;
use reqwest::Client;
use std::time::Duration;
use tokio::time::{sleep, Duration as TokioDuration};

/// Internal function to perform a single price fetch attempt
async fn fetch_price_once(
    client: &Client,
    url: &str,
    id: &str,
    vs: &str,
    pair: &str,
) -> Result<MarketTick, ConnectorError> {
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
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(ConnectorError::from_http_status(status.as_u16(), error_text));
    }

    let text = response
        .text()
        .await
        .map_err(|e| ConnectorError::Network(format!("Failed to read response body: {}", e)))?;

    let v: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| ConnectorError::Parse(format!("Invalid JSON: {}", e)))?;

    let price = v
        .get(id)
        .and_then(|o| o.get(vs))
        .and_then(|p| p.as_f64())
        .ok_or_else(|| {
            ConnectorError::Parse(format!("Could not extract price for {}/{}", id, vs))
        })?;

    Ok(MarketTick {
        exchange: "coingecko".to_string(),
        pair: pair.to_string(),
        bid: price,
        ask: price,
        ts: chrono::Utc::now().timestamp_millis() as u128,
    })
}

/// Run a polling loop to fetch prices from CoinGecko with automatic retry on failures.
///
/// This function implements exponential backoff retry logic for handling:
/// - Network connectivity issues
/// - Rate limiting (HTTP 429)
/// - Server errors (HTTP 5xx)
/// - Request timeouts
///
/// # Arguments
///
/// * `tx` - Channel sender for market ticks
/// * `pairs` - List of trading pairs (e.g., ["bitcoin/usd", "ethereum/usd"])
/// * `interval_ms` - Polling interval in milliseconds
///
/// # Returns
///
/// This function runs indefinitely and only returns on unrecoverable errors.
pub async fn run_coingecko_poll(
    mut tx: tokio::sync::mpsc::Sender<MarketTick>,
    pairs: Vec<String>,
    interval_ms: u64,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let client = Client::new();
    let interval = TokioDuration::from_millis(interval_ms);
    // Use conservative retry config for CoinGecko's rate limits
    let config = RetryConfig::conservative();

    loop {
        for pair in pairs.iter() {
            let parts: Vec<&str> = pair.split('/').collect();
            let (id, vs) = if parts.len() == 2 {
                (parts[0], parts[1])
            } else {
                (pair.as_str(), "usd")
            };
            let url = format!(
                "https://api.coingecko.com/api/v3/simple/price?ids={}&vs_currencies={}",
                id, vs
            );

            let result = with_retry(config.clone(), || {
                let client = &client;
                let url = &url;
                async move { fetch_price_once(client, url, id, vs, pair).await }
            })
            .await;

            match result {
                Ok(tick) => {
                    let _ = tx.send(tick).await;
                }
                Err(e) => {
                    info!("coingecko poll error for {} (after retries): {:?}", pair, e);
                }
            }
        }
        sleep(interval).await;
    }
}
