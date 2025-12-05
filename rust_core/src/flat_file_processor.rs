//! Fast flat file processing using Polars (< 1GB) and DataFusion (> 1GB)
//!
//! This module provides optimized Rust-based data processing for S3 flat files:
//! - Small files (< 1GB): Use Polars for fast in-memory processing
//! - Large files (> 1GB): Use Apache DataFusion for streaming/distributed processing
//! - S3 integration: Direct download from S3-compatible endpoints
//! - Format support: Parquet, CSV, JSON

use std::path::{Path, PathBuf};
use std::fs;
use serde::{Deserialize, Serialize};
use polars::prelude::*;

#[cfg(feature = "large-datasets")]
use datafusion::prelude::*;
#[cfg(feature = "large-datasets")]
use datafusion::arrow::array::RecordBatch;

/// File size threshold for switching between Polars and DataFusion
pub const SIZE_THRESHOLD_GB: f64 = 1.0;
pub const SIZE_THRESHOLD_BYTES: u64 = (SIZE_THRESHOLD_GB * 1024.0 * 1024.0 * 1024.0) as u64;

/// S3 credentials for flat file downloads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct S3Config {
    pub access_key_id: String,
    pub secret_access_key: String,
    pub endpoint: String,
    pub bucket: String,
    pub region: Option<String>,
}

/// Flat file processing result
#[derive(Debug, Serialize, Deserialize)]
pub struct FlatFileResult {
    pub success: bool,
    pub rows: usize,
    pub columns: usize,
    pub file_size_bytes: u64,
    pub processing_time_ms: u64,
    pub engine_used: String,
    pub message: String,
}

/// Download file from S3 and return local path
pub async fn download_from_s3(
    s3_config: &S3Config,
    s3_key: &str,
    local_path: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    use aws_config::BehaviorVersion;
    use aws_sdk_s3::Client;
    use aws_sdk_s3::config::{Credentials, Region};
    use tokio::io::AsyncWriteExt;

    // Create S3 client with custom credentials
    let creds = Credentials::new(
        &s3_config.access_key_id,
        &s3_config.secret_access_key,
        None,
        None,
        "massive-s3",
    );

    let region = Region::new(s3_config.region.clone().unwrap_or_else(|| "us-east-1".to_string()));

    let sdk_config = aws_config::defaults(BehaviorVersion::latest())
        .credentials_provider(creds)
        .region(region)
        .endpoint_url(&s3_config.endpoint)
        .load()
        .await;

    let client = Client::new(&sdk_config);

    // Download file
    let resp = client
        .get_object()
        .bucket(&s3_config.bucket)
        .key(s3_key)
        .send()
        .await?;

    // Create parent directory if needed
    if let Some(parent) = local_path.parent() {
        fs::create_dir_all(parent)?;
    }

    // Stream to file
    let mut file = tokio::fs::File::create(local_path).await?;
    let mut stream = resp.body;

    while let Some(chunk) = stream.try_next().await? {
        file.write_all(&chunk).await?;
    }

    file.flush().await?;
    Ok(local_path.to_path_buf())
}

/// Get file size in bytes
pub fn get_file_size(path: &Path) -> Result<u64, Box<dyn std::error::Error>> {
    let metadata = fs::metadata(path)?;
    Ok(metadata.len())
}

/// Process flat file using Polars (fast, in-memory, for files < 1GB)
pub fn process_with_polars(
    file_path: &Path,
    start_date: Option<&str>,
    end_date: Option<&str>,
    symbols: Option<Vec<String>>,
) -> Result<(DataFrame, FlatFileResult), Box<dyn std::error::Error>> {
    let start = std::time::Instant::now();
    let file_size = get_file_size(file_path)?;

    log::info!("🚀 Processing with Polars: {} ({:.2} MB)", 
               file_path.display(), 
               file_size as f64 / (1024.0 * 1024.0));

    // Read parquet file
    let mut df = LazyFrame::scan_parquet(file_path, Default::default())?;

    // Apply filters
    if let Some(start) = start_date {
        df = df.filter(col("timestamp").gt_eq(lit(start)));
    }

    if let Some(end) = end_date {
        df = df.filter(col("timestamp").lt_eq(lit(end)));
    }

    if let Some(syms) = symbols {
        // Filter by symbols using OR conditions
        let mut symbol_filter = col("symbol").eq(lit(syms[0].clone()));
        for sym in syms.iter().skip(1) {
            symbol_filter = symbol_filter.or(col("symbol").eq(lit(sym.clone())));
        }
        df = df.filter(symbol_filter);
    }

    // Execute query and collect
    let result = df.collect()?;
    let elapsed = start.elapsed();

    let result_info = FlatFileResult {
        success: true,
        rows: result.height(),
        columns: result.width(),
        file_size_bytes: file_size,
        processing_time_ms: elapsed.as_millis() as u64,
        engine_used: "Polars".to_string(),
        message: format!("Processed {} rows in {:.2}s", result.height(), elapsed.as_secs_f64()),
    };

    log::info!("✅ Polars: {} rows in {:.2}s", result.height(), elapsed.as_secs_f64());

    Ok((result, result_info))
}
/// Process flat file using Polars (works for all file sizes, optimized for < 1GB)
pub async fn process_flat_file_smart(
    file_path: &Path,
    start_date: Option<&str>,
    end_date: Option<&str>,
    symbols: Option<Vec<String>>,
) -> Result<FlatFileResult, Box<dyn std::error::Error>> {
    let file_size = get_file_size(file_path)?;
    let size_gb = file_size as f64 / (1024.0 * 1024.0 * 1024.0);

    log::info!("📦 File size {:.2} GB → Using Polars", size_gb);
    let (_df, result) = process_with_polars(file_path, start_date, end_date, symbols)?;
    Ok(result)
}

/// Download from S3 and process in one call
pub async fn download_and_process_s3(
    s3_config: &S3Config,
    s3_key: &str,
    local_dir: &Path,
    start_date: Option<&str>,
    end_date: Option<&str>,
    symbols: Option<Vec<String>>,
) -> Result<FlatFileResult, Box<dyn std::error::Error>> {
    // Generate local filename
    let filename = s3_key.split('/').last().unwrap_or("data.parquet");
    let local_path = local_dir.join(filename);

    log::info!("📥 Downloading from S3: {} → {}", s3_key, local_path.display());

    // Download
    let downloaded_path = download_from_s3(s3_config, s3_key, &local_path).await?;

    // Process
    process_flat_file_smart(&downloaded_path, start_date, end_date, symbols).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_threshold() {
        assert_eq!(SIZE_THRESHOLD_BYTES, 1_073_741_824); // 1 GB
    }

    #[test]
    fn test_file_size_check() {
        let small_size = 500_000_000; // 500 MB
        let large_size = 2_000_000_000; // 2 GB

        assert!(small_size < SIZE_THRESHOLD_BYTES);
        assert!(large_size > SIZE_THRESHOLD_BYTES);
    }
}
