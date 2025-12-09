"""
Data Persistence Module
========================

Handles saving, loading, and appending market data to disk storage.
Uses Parquet format for efficient storage and fast I/O.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data directory paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
PERSISTED_DATA_DIR = DATA_DIR / "persisted"
METADATA_FILE = PERSISTED_DATA_DIR / "metadata.json"

# Ensure directories exist
PERSISTED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _get_dataset_filename(dataset_name: str) -> Path:
    """
    Generate a filename for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Path to the dataset file
    """
    safe_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in dataset_name)
    return PERSISTED_DATA_DIR / f"{safe_name}.parquet"


def _load_metadata() -> Dict:
    """
    Load metadata about persisted datasets.
    
    Returns:
        Dictionary containing metadata for all datasets
    """
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
            return {}
    return {}


def _save_metadata(metadata: Dict):
    """
    Save metadata about persisted datasets.
    
    Args:
        metadata: Dictionary containing metadata for all datasets
    """
    try:
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")


def save_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    symbols: List[str],
    source: str,
    date_range: Optional[Tuple] = None,
    append: bool = False
) -> bool:
    """
    Save a dataset to disk. Can append to existing dataset or replace it.
    
    Args:
        df: DataFrame containing market data
        dataset_name: Name for the dataset (used as filename)
        symbols: List of symbols in the dataset
        source: Data source (e.g., 'Finnhub', 'Binance')
        date_range: Tuple of (start_date, end_date) or None
        append: If True, append to existing dataset; if False, replace
        
    Returns:
        True if successful, False otherwise
    """
    try:
        filepath = _get_dataset_filename(dataset_name)
        metadata = _load_metadata()
        
        # If appending and file exists, load and combine
        if append and filepath.exists():
            try:
                existing_df = pd.read_parquet(filepath)
                logger.info(f"Loading existing dataset: {len(existing_df)} rows")
                
                # Combine dataframes and remove duplicates
                df = pd.concat([existing_df, df], ignore_index=True)
                
                # Remove duplicates based on symbol and timestamp if available
                if 'symbol' in df.columns and 'timestamp' in df.columns:
                    df = df.drop_duplicates(subset=['symbol', 'timestamp'], keep='last')
                elif 'symbol' in df.columns:
                    df = df.drop_duplicates(subset=['symbol'], keep='last')
                
                logger.info(f"Combined dataset: {len(df)} rows after deduplication")
                
                # Merge symbols list
                existing_symbols = metadata.get(dataset_name, {}).get('symbols', [])
                symbols = list(set(existing_symbols + symbols))
                
            except Exception as e:
                logger.warning(f"Failed to load existing dataset for appending: {e}")
        
        # Save the dataframe
        df.to_parquet(filepath, index=False, compression='snappy')
        logger.info(f"Saved {len(df)} rows to {filepath}")
        
        # Update metadata
        if dataset_name not in metadata:
            metadata[dataset_name] = {}
        
        metadata[dataset_name].update({
            'symbols': sorted(symbols),
            'source': source,
            'date_range': date_range,
            'row_count': len(df),
            'columns': list(df.columns),
            'last_updated': datetime.now().isoformat(),
            'filepath': str(filepath.relative_to(DATA_DIR))
        })
        
        _save_metadata(metadata)
        logger.info(f"Updated metadata for dataset '{dataset_name}'")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to save dataset '{dataset_name}': {e}")
        return False


def load_dataset(dataset_name: str) -> Optional[tuple[pd.DataFrame, Dict]]:
    """
    Load a persisted dataset from disk along with its metadata.
    
    Args:
        dataset_name: Name of the dataset to load
        
    Returns:
        Tuple of (DataFrame, metadata dict) if found, None otherwise
    """
    try:
        filepath = _get_dataset_filename(dataset_name)
        
        if not filepath.exists():
            logger.warning(f"Dataset '{dataset_name}' not found")
            return None
        
        df = pd.read_parquet(filepath)
        
        # Load metadata for this dataset
        metadata = _load_metadata()
        dataset_meta = metadata.get(dataset_name, {})
        
        logger.info(f"Loaded dataset '{dataset_name}': {len(df)} rows")
        return df, dataset_meta
        
    except Exception as e:
        logger.error(f"Failed to load dataset '{dataset_name}': {e}")
        return None


def load_all_datasets() -> Dict[str, pd.DataFrame]:
    """
    Load all persisted datasets from disk.
    
    Returns:
        Dictionary mapping dataset names to DataFrames
    """
    metadata = _load_metadata()
    datasets = {}
    
    for dataset_name in metadata.keys():
        result = load_dataset(dataset_name)
        if result is not None:
            df, _ = result
            datasets[dataset_name] = df
    
    logger.info(f"Loaded {len(datasets)} datasets from disk")
    return datasets


def get_dataset_metadata(dataset_name: Optional[str] = None) -> Dict:
    """
    Get metadata about persisted dataset(s).
    
    Args:
        dataset_name: Name of specific dataset, or None for all datasets
        
    Returns:
        Metadata dictionary for the dataset or all datasets
    """
    metadata = _load_metadata()
    
    if dataset_name:
        return metadata.get(dataset_name, {})
    
    return metadata


def delete_dataset(dataset_name: str) -> bool:
    """
    Delete a persisted dataset.
    
    Args:
        dataset_name: Name of the dataset to delete
        
    Returns:
        True if successful, False otherwise
    """
    try:
        filepath = _get_dataset_filename(dataset_name)
        
        if filepath.exists():
            filepath.unlink()
            logger.info(f"Deleted dataset file: {filepath}")
        
        # Update metadata
        metadata = _load_metadata()
        if dataset_name in metadata:
            del metadata[dataset_name]
            _save_metadata(metadata)
            logger.info(f"Removed metadata for dataset '{dataset_name}'")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to delete dataset '{dataset_name}': {e}")
        return False


def list_datasets() -> List[Dict]:
    """
    List all persisted datasets with their metadata.
    
    Returns:
        List of dictionaries containing dataset information
    """
    metadata = _load_metadata()
    
    datasets = []
    for name, info in metadata.items():
        datasets.append({
            'name': name,
            'symbols': info.get('symbols', []),
            'source': info.get('source', 'Unknown'),
            'row_count': info.get('row_count', 0),
            'last_updated': info.get('last_updated', 'Unknown'),
            'date_range': info.get('date_range')
        })
    
    # Sort by last updated (most recent first)
    datasets.sort(key=lambda x: x.get('last_updated', ''), reverse=True)
    
    return datasets


def get_storage_stats() -> Dict:
    """
    Get statistics about persisted data storage.
    
    Returns:
        Dictionary containing storage statistics
    """
    metadata = _load_metadata()
    
    total_datasets = len(metadata)
    total_rows = sum(info.get('row_count', 0) for info in metadata.values())
    
    # Calculate total storage size
    total_size_bytes = sum(
        _get_dataset_filename(name).stat().st_size 
        for name in metadata.keys() 
        if _get_dataset_filename(name).exists()
    )
    
    total_size_mb = total_size_bytes / (1024 * 1024)
    
    # Get all unique symbols
    all_symbols = set()
    for info in metadata.values():
        all_symbols.update(info.get('symbols', []))
    
    return {
        'total_datasets': total_datasets,
        'total_rows': total_rows,
        'total_symbols': len(all_symbols),
        'total_size_mb': round(total_size_mb, 2),
        'storage_path': str(PERSISTED_DATA_DIR)
    }


def merge_datasets(dataset_names: List[str], new_dataset_name: str) -> bool:
    """
    Merge multiple datasets into a single new dataset.
    
    Args:
        dataset_names: List of dataset names to merge
        new_dataset_name: Name for the merged dataset
        
    Returns:
        True if successful, False otherwise
    """
    try:
        dfs = []
        all_symbols = []
        sources = []
        
        for name in dataset_names:
            result = load_dataset(name)
            if result is not None:
                df, meta = result
                dfs.append(df)
                
                # Use metadata from load_dataset
                all_symbols.extend(meta.get('symbols', []))
                sources.append(meta.get('source', 'Unknown'))
        
        if not dfs:
            logger.warning("No datasets found to merge")
            return False
        
        # Combine all dataframes
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # Remove duplicates
        if 'symbol' in merged_df.columns and 'timestamp' in merged_df.columns:
            merged_df = merged_df.drop_duplicates(subset=['symbol', 'timestamp'], keep='last')
        
        # Get unique symbols and sources
        unique_symbols = list(set(all_symbols))
        source = ", ".join(set(sources))
        
        # Save merged dataset
        return save_dataset(
            merged_df,
            new_dataset_name,
            unique_symbols,
            source,
            append=False
        )
        
    except Exception as e:
        logger.error(f"Failed to merge datasets: {e}")
        return False
