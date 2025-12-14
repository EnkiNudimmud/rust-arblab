use reqwest::Client;
use serde_json::Value;
use std::env;

pub async fn fetch_alpaca_bars(
    api_key: &str,
    api_secret: &str,
    base_url: &str,
    symbol: &str,
    start: &str,
    end: &str,
    timeframe: &str,
    limit: u32,
) -> Result<Value, reqwest::Error> {
    let url = format!("{}/v2/stocks/{}/bars", base_url, symbol);
    let client = Client::new();
    let resp = client
        .get(&url)
        .query(&[
            ("start", start),
            ("end", end),
            ("timeframe", timeframe),
            ("limit", &limit.to_string()),
        ])
        .header("APCA-API-KEY-ID", api_key)
        .header("APCA-API-SECRET-KEY", api_secret)
        .send()
        .await?
        .json::<Value>()
        .await?;
    Ok(resp)
}

// WebSocket streaming for live trading (example)
use tokio_tungstenite::connect_async;
use url::Url;

pub async fn stream_alpaca_quotes(api_key: &str, api_secret: &str, symbols: &[&str]) {
    let ws_url = "wss://stream.data.alpaca.markets/v2/sip";
    let url = Url::parse(ws_url).unwrap();
    let (ws_stream, _) = connect_async(url).await.expect("Failed to connect");
    // Authenticate and subscribe logic goes here
    // See Alpaca docs for message format
}
