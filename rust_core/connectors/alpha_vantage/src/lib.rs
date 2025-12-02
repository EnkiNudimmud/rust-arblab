// Alpha Vantage REST API Connector
// =================================
// Implements Alpha Vantage API for stocks, forex, and crypto
// Free tier: 25 API requests per day, 5 requests per minute
// 
// Supported endpoints:
// - TIME_SERIES_INTRADAY: 1min, 5min, 15min, 30min, 60min intervals
// - TIME_SERIES_DAILY: Daily prices
// - QUOTE_ENDPOINT: Real-time quote
// - FX_INTRADAY: Forex intraday
// - DIGITAL_CURRENCY_DAILY: Crypto daily prices

use connectors_common::types::MarketTick;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use tokio::sync::mpsc::Sender;
use chrono::{DateTime, NaiveDateTime, Utc};

const BASE_URL: &str = "https://www.alphavantage.co/query";

// Free tier rate limits
pub const FREE_TIER_DAILY_LIMIT: u32 = 25;
pub const FREE_TIER_PER_MINUTE_LIMIT: u32 = 5;

#[derive(Debug, Clone)]
pub struct AlphaVantageConfig {
    pub api_key: String,
    pub base_url: String,
}

impl Default for AlphaVantageConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: BASE_URL.to_string(),
        }
    }
}

impl AlphaVantageConfig {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            base_url: BASE_URL.to_string(),
        }
    }
}

// Time series data structures
#[derive(Debug, Deserialize, Serialize)]
pub struct TimeSeriesData {
    #[serde(rename = "Meta Data")]
    pub metadata: Metadata,
    #[serde(flatten)]
    pub time_series: HashMap<String, HashMap<String, PriceData>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Metadata {
    #[serde(rename = "1. Information")]
    pub information: String,
    #[serde(rename = "2. Symbol")]
    pub symbol: String,
    #[serde(rename = "3. Last Refreshed")]
    pub last_refreshed: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct PriceData {
    #[serde(rename = "1. open")]
    pub open: String,
    #[serde(rename = "2. high")]
    pub high: String,
    #[serde(rename = "3. low")]
    pub low: String,
    #[serde(rename = "4. close")]
    pub close: String,
    #[serde(rename = "5. volume")]
    pub volume: String,
}

// Quote endpoint structure
#[derive(Debug, Deserialize, Serialize)]
pub struct QuoteResponse {
    #[serde(rename = "Global Quote")]
    pub global_quote: GlobalQuote,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct GlobalQuote {
    #[serde(rename = "01. symbol")]
    pub symbol: String,
    #[serde(rename = "02. open")]
    pub open: String,
    #[serde(rename = "03. high")]
    pub high: String,
    #[serde(rename = "04. low")]
    pub low: String,
    #[serde(rename = "05. price")]
    pub price: String,
    #[serde(rename = "06. volume")]
    pub volume: String,
    #[serde(rename = "07. latest trading day")]
    pub latest_trading_day: String,
    #[serde(rename = "08. previous close")]
    pub previous_close: String,
    #[serde(rename = "09. change")]
    pub change: String,
    #[serde(rename = "10. change percent")]
    pub change_percent: String,
}

// Forex structures
#[derive(Debug, Deserialize, Serialize)]
pub struct ForexData {
    #[serde(rename = "Meta Data")]
    pub metadata: ForexMetadata,
    #[serde(flatten)]
    pub time_series: HashMap<String, HashMap<String, ForexPrice>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ForexMetadata {
    #[serde(rename = "1. Information")]
    pub information: String,
    #[serde(rename = "2. From Symbol")]
    pub from_symbol: String,
    #[serde(rename = "3. To Symbol")]
    pub to_symbol: String,
    #[serde(rename = "4. Last Refreshed")]
    pub last_refreshed: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ForexPrice {
    #[serde(rename = "1. open")]
    pub open: String,
    #[serde(rename = "2. high")]
    pub high: String,
    #[serde(rename = "3. low")]
    pub low: String,
    #[serde(rename = "4. close")]
    pub close: String,
}

// Crypto structures
#[derive(Debug, Deserialize, Serialize)]
pub struct CryptoData {
    #[serde(rename = "Meta Data")]
    pub metadata: CryptoMetadata,
    #[serde(flatten)]
    pub time_series: HashMap<String, HashMap<String, CryptoPrice>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CryptoMetadata {
    #[serde(rename = "1. Information")]
    pub information: String,
    #[serde(rename = "2. Digital Currency Code")]
    pub digital_currency_code: String,
    #[serde(rename = "3. Digital Currency Name")]
    pub digital_currency_name: String,
    #[serde(rename = "4. Market Code")]
    pub market_code: String,
    #[serde(rename = "5. Last Refreshed")]
    pub last_refreshed: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CryptoPrice {
    #[serde(rename = "1a. open (USD)")]
    pub open_usd: String,
    #[serde(rename = "2a. high (USD)")]
    pub high_usd: String,
    #[serde(rename = "3a. low (USD)")]
    pub low_usd: String,
    #[serde(rename = "4a. close (USD)")]
    pub close_usd: String,
    #[serde(rename = "5. volume")]
    pub volume: String,
    #[serde(rename = "6. market cap (USD)")]
    pub market_cap_usd: String,
}

/// Fetch intraday time series data
/// Interval: 1min, 5min, 15min, 30min, 60min
pub async fn fetch_intraday(
    config: &AlphaVantageConfig,
    symbol: &str,
    interval: &str,
) -> Result<TimeSeriesData, Box<dyn Error + Send + Sync>> {
    let client = Client::new();
    let url = format!(
        "{}?function=TIME_SERIES_INTRADAY&symbol={}&interval={}&apikey={}",
        config.base_url, symbol, interval, config.api_key
    );

    let response = client.get(&url).send().await?;
    let data: TimeSeriesData = response.json().await?;
    
    Ok(data)
}

/// Fetch daily time series data
pub async fn fetch_daily(
    config: &AlphaVantageConfig,
    symbol: &str,
) -> Result<TimeSeriesData, Box<dyn Error + Send + Sync>> {
    let client = Client::new();
    let url = format!(
        "{}?function=TIME_SERIES_DAILY&symbol={}&apikey={}",
        config.base_url, symbol, config.api_key
    );

    let response = client.get(&url).send().await?;
    let data: TimeSeriesData = response.json().await?;
    
    Ok(data)
}

/// Fetch real-time quote
pub async fn fetch_quote(
    config: &AlphaVantageConfig,
    symbol: &str,
) -> Result<QuoteResponse, Box<dyn Error + Send + Sync>> {
    let client = Client::new();
    let url = format!(
        "{}?function=GLOBAL_QUOTE&symbol={}&apikey={}",
        config.base_url, symbol, config.api_key
    );

    let response = client.get(&url).send().await?;
    let data: QuoteResponse = response.json().await?;
    
    Ok(data)
}

/// Fetch forex intraday data
pub async fn fetch_forex_intraday(
    config: &AlphaVantageConfig,
    from_symbol: &str,
    to_symbol: &str,
    interval: &str,
) -> Result<ForexData, Box<dyn Error + Send + Sync>> {
    let client = Client::new();
    let url = format!(
        "{}?function=FX_INTRADAY&from_symbol={}&to_symbol={}&interval={}&apikey={}",
        config.base_url, from_symbol, to_symbol, interval, config.api_key
    );

    let response = client.get(&url).send().await?;
    let data: ForexData = response.json().await?;
    
    Ok(data)
}

/// Fetch crypto daily data
pub async fn fetch_crypto_daily(
    config: &AlphaVantageConfig,
    symbol: &str,
    market: &str,
) -> Result<CryptoData, Box<dyn Error + Send + Sync>> {
    let client = Client::new();
    let url = format!(
        "{}?function=DIGITAL_CURRENCY_DAILY&symbol={}&market={}&apikey={}",
        config.base_url, symbol, market, config.api_key
    );

    let response = client.get(&url).send().await?;
    let data: CryptoData = response.json().await?;
    
    Ok(data)
}

/// Convert quote to MarketTick
fn quote_to_tick(quote: &GlobalQuote) -> Result<MarketTick, Box<dyn Error + Send + Sync>> {
    let price: f64 = quote.price.parse()?;
    let volume: f64 = quote.volume.parse()?;
    
    Ok(MarketTick {
        symbol: quote.symbol.clone(),
        exchange: "ALPHA_VANTAGE".to_string(),
        price,
        quantity: volume,
        side: "UNKNOWN".to_string(),
        timestamp_ms: chrono::Utc::now().timestamp_millis() as u64,
    })
}

/// Poll Alpha Vantage for multiple symbols
/// Respects free tier rate limits (5 requests/minute)
pub async fn poll_quotes(
    config: AlphaVantageConfig,
    tx: Sender<MarketTick>,
    symbols: Vec<String>,
    interval_secs: u64,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let delay_between_requests = std::time::Duration::from_secs(12); // 5 per minute max
    
    loop {
        for symbol in &symbols {
            match fetch_quote(&config, symbol).await {
                Ok(quote_response) => {
                    match quote_to_tick(&quote_response.global_quote) {
                        Ok(tick) => {
                            if let Err(e) = tx.send(tick).await {
                                log::error!("Failed to send tick: {}", e);
                            }
                        }
                        Err(e) => {
                            log::error!("Failed to convert quote to tick: {}", e);
                        }
                    }
                }
                Err(e) => {
                    log::error!("Failed to fetch quote for {}: {}", symbol, e);
                }
            }
            
            // Rate limiting: wait between requests
            tokio::time::sleep(delay_between_requests).await;
        }
        
        // Wait before next polling cycle
        tokio::time::sleep(std::time::Duration::from_secs(interval_secs)).await;
    }
}

/// Convert time series data to MarketTicks
pub fn time_series_to_ticks(
    data: &TimeSeriesData,
    max_points: Option<usize>,
) -> Vec<MarketTick> {
    let mut ticks = Vec::new();
    
    // Extract time series data (key varies by function)
    if let Some(series) = data.time_series.values().next() {
        let mut entries: Vec<_> = series.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0)); // Sort by timestamp
        
        if let Some(max) = max_points {
            entries.truncate(max);
        }
        
        for (timestamp, price) in entries {
            if let (Ok(close), Ok(volume)) = (price.close.parse::<f64>(), price.volume.parse::<f64>()) {
                // Parse timestamp
                let ts_ms = parse_timestamp(timestamp).unwrap_or(0);
                
                ticks.push(MarketTick {
                    symbol: data.metadata.symbol.clone(),
                    exchange: "ALPHA_VANTAGE".to_string(),
                    price: close,
                    quantity: volume,
                    side: "TRADE".to_string(),
                    timestamp_ms: ts_ms,
                });
            }
        }
    }
    
    ticks
}

/// Parse Alpha Vantage timestamp to milliseconds
fn parse_timestamp(ts: &str) -> Result<u64, chrono::ParseError> {
    // Try parsing with time first (e.g., "2024-12-02 16:00:00")
    if let Ok(dt) = NaiveDateTime::parse_from_str(ts, "%Y-%m-%d %H:%M:%S") {
        let utc: DateTime<Utc> = DateTime::from_naive_utc_and_offset(dt, Utc);
        return Ok(utc.timestamp_millis() as u64);
    }
    
    // Try date only (e.g., "2024-12-02")
    if let Ok(date) = chrono::NaiveDate::parse_from_str(ts, "%Y-%m-%d") {
        let dt = date.and_hms_opt(0, 0, 0).unwrap();
        let utc: DateTime<Utc> = DateTime::from_naive_utc_and_offset(dt, Utc);
        return Ok(utc.timestamp_millis() as u64);
    }
    
    Err(chrono::ParseError(chrono::format::ParseErrorKind::Invalid))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_timestamp_with_time() {
        let ts = "2024-12-02 16:00:00";
        let result = parse_timestamp(ts);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_timestamp_date_only() {
        let ts = "2024-12-02";
        let result = parse_timestamp(ts);
        assert!(result.is_ok());
    }

    #[test]
    fn test_config_creation() {
        let config = AlphaVantageConfig::new("test_key".to_string());
        assert_eq!(config.api_key, "test_key");
        assert_eq!(config.base_url, BASE_URL);
    }
}
