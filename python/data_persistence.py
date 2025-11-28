# python/data_persistence.py
"""
Data persistence module for long-living sessions.

Provides functionality to:
- Save fetched data to disk in the /data folder
- Load persisted data on session restart
- Stack/merge new data with existing data
- Manage saved datasets (list, delete, rename)
"""

import os
import json
import re
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Default data directory
DATA_DIR = Path(__file__).parent.parent / "data" / "saved_datasets"


def get_data_dir() -> Path:
    """Get the data directory path, creating it if necessary."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def sanitize_name(name: str) -> str:
    """
    Sanitize a dataset name to prevent directory traversal and filesystem issues.
    
    Args:
        name: Raw dataset name
        
    Returns:
        Sanitized name safe for use in filesystem paths
        
    Raises:
        ValueError: If name is empty or contains only invalid characters
    """
    # Remove any path separators and parent directory references
    safe_name = name.replace('/', '_').replace('\\', '_').replace('..', '_')
    
    # Only allow alphanumeric, underscore, and hyphen
    safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', safe_name)
    
    # Remove leading/trailing underscores and collapse multiple underscores
    safe_name = re.sub(r'_+', '_', safe_name).strip('_')
    
    if not safe_name:
        raise ValueError("Dataset name cannot be empty or contain only invalid characters")
    
    return safe_name


def generate_dataset_name(symbols: List[str], interval: str, source: str) -> str:
    """Generate a unique dataset name based on parameters."""
    # Use first 3 symbols and count for naming
    symbol_part = "_".join(symbols[:3])
    if len(symbols) > 3:
        symbol_part += f"_+{len(symbols)-3}more"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_name = f"{source}_{symbol_part}_{interval}_{timestamp}"
    return sanitize_name(raw_name)


def save_dataset(
    df: pd.DataFrame,
    name: str,
    metadata: Optional[Dict] = None
) -> str:
    """
    Save a dataset to disk with metadata.
    
    Args:
        df: DataFrame to save
        name: Dataset name
        metadata: Optional metadata dict (symbols, interval, source, date_range, etc.)
    
    Returns:
        Path to saved dataset
    """
    data_dir = get_data_dir()
    
    # Sanitize name to prevent directory traversal
    safe_name = sanitize_name(name)
    
    # Create dataset directory
    dataset_dir = data_dir / safe_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Save data as parquet (efficient for time series)
    data_path = dataset_dir / "data.parquet"
    
    # Reset index if MultiIndex for easier storage
    df_to_save = df.reset_index() if isinstance(df.index, pd.MultiIndex) else df
    df_to_save.to_parquet(data_path, index=False)
    
    # Save metadata
    meta = metadata or {}
    meta.update({
        "name": name,
        "created_at": datetime.now().isoformat(),
        "rows": len(df),
        "columns": list(df.columns) if not isinstance(df.index, pd.MultiIndex) else list(df.reset_index().columns),
    })
    
    # Extract symbols from data
    if "symbol" in df_to_save.columns:
        meta["symbols"] = df_to_save["symbol"].unique().tolist()
    
    # Extract date range
    if "timestamp" in df_to_save.columns:
        meta["date_range"] = {
            "start": str(df_to_save["timestamp"].min()),
            "end": str(df_to_save["timestamp"].max())
        }
    
    meta_path = dataset_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    
    logger.info(f"Saved dataset '{name}' with {len(df)} rows to {dataset_dir}")
    return str(dataset_dir)


def load_dataset(name: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Load a dataset from disk.
    
    Args:
        name: Dataset name
    
    Returns:
        Tuple of (DataFrame, metadata dict)
    """
    data_dir = get_data_dir()
    
    # Sanitize name to prevent directory traversal
    safe_name = sanitize_name(name)
    dataset_dir = data_dir / safe_name
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset '{name}' not found in {data_dir}")
    
    # Load data
    data_path = dataset_dir / "data.parquet"
    df = pd.read_parquet(data_path)
    
    # Convert timestamp to datetime if present
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Restore MultiIndex if we have timestamp and symbol
    if "timestamp" in df.columns and "symbol" in df.columns:
        df = df.set_index(["timestamp", "symbol"]).sort_index()
    
    # Load metadata
    meta_path = dataset_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {"name": name}
    
    return df, metadata


def list_datasets() -> List[Dict]:
    """
    List all saved datasets with their metadata.
    
    Returns:
        List of metadata dicts for each dataset
    """
    data_dir = get_data_dir()
    datasets = []
    
    for item in data_dir.iterdir():
        if item.is_dir():
            meta_path = item / "metadata.json"
            if meta_path.exists():
                try:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    meta["path"] = str(item)
                    datasets.append(meta)
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {item}: {e}")
    
    # Sort by creation date (newest first)
    datasets.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return datasets


def delete_dataset(name: str) -> bool:
    """
    Delete a saved dataset.
    
    Args:
        name: Dataset name
    
    Returns:
        True if deleted successfully
    """
    import shutil
    
    data_dir = get_data_dir()
    
    # Sanitize name to prevent directory traversal
    safe_name = sanitize_name(name)
    dataset_dir = data_dir / safe_name
    
    if not dataset_dir.exists():
        return False
    
    # Additional safety check: ensure the path is within data_dir
    try:
        dataset_dir.resolve().relative_to(data_dir.resolve())
    except ValueError:
        logger.error(f"Security: Attempted to delete outside data directory: {name}")
        return False
    
    try:
        shutil.rmtree(dataset_dir)
        logger.info(f"Deleted dataset '{name}'")
        return True
    except Exception as e:
        logger.error(f"Failed to delete dataset '{name}': {e}")
        return False


def merge_dataframes(
    existing: pd.DataFrame,
    new: pd.DataFrame,
    mode: str = "append"
) -> pd.DataFrame:
    """
    Merge two DataFrames with different strategies.
    
    Args:
        existing: Existing DataFrame
        new: New DataFrame to merge
        mode: 'append' (add new, keep existing), 'update' (prefer new for overlaps),
              'replace' (discard existing)
    
    Returns:
        Merged DataFrame
    """
    if mode == "replace":
        return new
    
    # Ensure both have consistent format
    if isinstance(existing.index, pd.MultiIndex):
        existing = existing.reset_index()
    if isinstance(new.index, pd.MultiIndex):
        new = new.reset_index()
    
    if mode == "append":
        # Concatenate and remove duplicates (keep existing)
        combined = pd.concat([existing, new], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["timestamp", "symbol"],
            keep="first"  # Keep existing data
        )
    elif mode == "update":
        # Concatenate and remove duplicates (prefer new)
        combined = pd.concat([new, existing], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["timestamp", "symbol"],
            keep="first"  # Keep new data
        )
    else:
        combined = pd.concat([existing, new], ignore_index=True)
    
    # Sort by timestamp and symbol
    combined = combined.sort_values(["timestamp", "symbol"])
    
    # Restore MultiIndex
    combined = combined.set_index(["timestamp", "symbol"]).sort_index()
    
    return combined


def stack_data(
    session_data: Optional[pd.DataFrame],
    new_data: pd.DataFrame,
    mode: str = "append"
) -> pd.DataFrame:
    """
    Stack new data with existing session data.
    
    Args:
        session_data: Current session data (may be None)
        new_data: New data to add
        mode: Merge mode ('append', 'update', 'replace')
    
    Returns:
        Combined DataFrame
    """
    if session_data is None or len(session_data) == 0:
        return new_data
    
    return merge_dataframes(session_data, new_data, mode)


def get_dataset_info(name: str) -> Optional[Dict]:
    """Get metadata for a specific dataset."""
    data_dir = get_data_dir()
    
    # Sanitize name to prevent directory traversal
    safe_name = sanitize_name(name)
    meta_path = data_dir / safe_name / "metadata.json"
    
    if meta_path.exists():
        with open(meta_path, "r") as f:
            return json.load(f)
    return None


def export_dataset_csv(name: str, output_path: Optional[str] = None) -> str:
    """
    Export a dataset to CSV format.
    
    Args:
        name: Dataset name
        output_path: Optional output path (defaults to data directory)
    
    Returns:
        Path to exported CSV
    """
    # load_dataset already sanitizes the name
    df, meta = load_dataset(name)
    
    # Reset index for CSV export
    df_export = df.reset_index() if isinstance(df.index, pd.MultiIndex) else df
    
    if output_path is None:
        data_dir = get_data_dir()
        # Sanitize name for the output filename
        safe_name = sanitize_name(name)
        output_path = str(data_dir / f"{safe_name}.csv")
    
    df_export.to_csv(output_path, index=False)
    return output_path


def get_total_storage_size() -> int:
    """Get total storage size of all saved datasets in bytes."""
    data_dir = get_data_dir()
    total_size = 0
    
    for item in data_dir.rglob("*"):
        if item.is_file():
            total_size += item.stat().st_size
    
    return total_size


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
