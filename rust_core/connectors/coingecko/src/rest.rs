use connectors_common::errors::ConnectorError;
use connectors_common::retry::{with_retry, RetryConfig};
use connectors_common::types::MarketTick;
use log::info;
use reqwest::Client;
use std::time::Duration;
use tokio::time::{sleep, Duration as TokioDuration};

pub async fn run_coingecko_poll(tx: tokio::sync::mpsc::Sender<MarketTick>, pairs: Vec<String>, interval_ms: u64) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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
