# tests/test_data_persistence.py
"""
Tests for the data persistence module.
"""

import os
import tempfile
import shutil
import unittest
import pandas as pd
import numpy as np
from datetime import datetime

# Adjust path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.data_persistence import (
    save_dataset, load_dataset, list_datasets, delete_dataset,
    merge_dataframes, stack_data, generate_dataset_name,
    format_size, get_data_dir
)


class TestDataPersistence(unittest.TestCase):
    """Tests for data persistence functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Create a temporary directory for test data
        cls.test_dir = tempfile.mkdtemp()
        
        # Monkey-patch the data directory for testing
        import python.data_persistence as dp
        cls.original_data_dir = dp.DATA_DIR
        dp.DATA_DIR = Path(cls.test_dir) / "saved_datasets"
        dp.DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        # Restore original data directory
        import python.data_persistence as dp
        dp.DATA_DIR = cls.original_data_dir
        
        # Remove temporary directory
        shutil.rmtree(cls.test_dir)
    
    def create_test_dataframe(self, symbols=None, rows=100):
        """Create a test DataFrame with OHLCV data."""
        if symbols is None:
            symbols = ["BTC/USDT", "ETH/USDT"]
        
        data = []
        for symbol in symbols:
            for i in range(rows):
                data.append({
                    "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=i),
                    "symbol": symbol,
                    "open": 100 + np.random.randn(),
                    "high": 101 + np.random.randn(),
                    "low": 99 + np.random.randn(),
                    "close": 100.5 + np.random.randn(),
                    "volume": 1000 + np.random.randint(0, 500)
                })
        
        df = pd.DataFrame(data)
        df = df.set_index(["timestamp", "symbol"]).sort_index()
        return df
    
    def test_save_and_load_dataset(self):
        """Test saving and loading a dataset."""
        df = self.create_test_dataframe()
        name = "test_save_load"
        
        # Save
        path = save_dataset(df, name, {"source": "test", "interval": "1m"})
        self.assertTrue(os.path.exists(path))
        
        # Load
        loaded_df, metadata = load_dataset(name)
        
        # Check data integrity
        self.assertEqual(len(loaded_df), len(df))
        self.assertIn("source", metadata)
        self.assertEqual(metadata["source"], "test")
        
        # Cleanup
        delete_dataset(name)
    
    def test_list_datasets(self):
        """Test listing saved datasets."""
        # Save a few datasets
        df = self.create_test_dataframe()
        names = ["test_list_1", "test_list_2"]
        
        for name in names:
            save_dataset(df, name)
        
        # List datasets
        datasets = list_datasets()
        dataset_names = [d["name"] for d in datasets]
        
        for name in names:
            self.assertIn(name, dataset_names)
        
        # Cleanup
        for name in names:
            delete_dataset(name)
    
    def test_delete_dataset(self):
        """Test deleting a dataset."""
        df = self.create_test_dataframe()
        name = "test_delete"
        
        save_dataset(df, name)
        datasets_before = list_datasets()
        
        result = delete_dataset(name)
        self.assertTrue(result)
        
        datasets_after = list_datasets()
        self.assertEqual(len(datasets_after), len(datasets_before) - 1)
    
    def test_merge_dataframes_append(self):
        """Test merging DataFrames in append mode."""
        # Create two overlapping DataFrames
        df1 = self.create_test_dataframe(symbols=["BTC/USDT"], rows=50)
        df2 = self.create_test_dataframe(symbols=["BTC/USDT"], rows=50)
        
        # Reset indices for merging
        df1_reset = df1.reset_index()
        df2_reset = df2.reset_index()
        
        # Shift df2 timestamps to create overlap
        df2_reset["timestamp"] = df2_reset["timestamp"] + pd.Timedelta(minutes=25)
        df2 = df2_reset.set_index(["timestamp", "symbol"])
        
        # Merge in append mode (keep existing for overlaps)
        merged = merge_dataframes(df1, df2, mode="append")
        
        # Should have unique timestamps
        merged_reset = merged.reset_index()
        unique_timestamps = merged_reset.drop_duplicates(subset=["timestamp", "symbol"])
        self.assertEqual(len(merged_reset), len(unique_timestamps))
    
    def test_merge_dataframes_update(self):
        """Test merging DataFrames in update mode."""
        df1 = self.create_test_dataframe(symbols=["ETH/USDT"], rows=30)
        df2 = self.create_test_dataframe(symbols=["ETH/USDT"], rows=30)
        
        # Merge in update mode (prefer new for overlaps)
        merged = merge_dataframes(df1, df2, mode="update")
        
        # Should have data
        self.assertGreater(len(merged), 0)
    
    def test_merge_dataframes_replace(self):
        """Test merging DataFrames in replace mode."""
        df1 = self.create_test_dataframe(symbols=["BTC/USDT"], rows=100)
        df2 = self.create_test_dataframe(symbols=["ETH/USDT"], rows=50)
        
        # Replace mode should discard existing
        merged = merge_dataframes(df1, df2, mode="replace")
        
        self.assertEqual(len(merged), len(df2))
    
    def test_stack_data_with_none(self):
        """Test stacking data when existing is None."""
        df = self.create_test_dataframe()
        
        result = stack_data(None, df, mode="append")
        self.assertEqual(len(result), len(df))
    
    def test_stack_data_append(self):
        """Test stacking data in append mode."""
        df1 = self.create_test_dataframe(symbols=["BTC/USDT"], rows=50)
        df2 = self.create_test_dataframe(symbols=["SOL/USDT"], rows=50)
        
        result = stack_data(df1, df2, mode="append")
        
        # Should have data from both symbols
        result_reset = result.reset_index()
        symbols = result_reset["symbol"].unique()
        self.assertIn("BTC/USDT", symbols)
        self.assertIn("SOL/USDT", symbols)
    
    def test_generate_dataset_name(self):
        """Test dataset name generation."""
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        name = generate_dataset_name(symbols, "1m", "ccxt")
        
        self.assertIn("ccxt", name)
        self.assertIn("1m", name)
        self.assertIn("BTC", name)
    
    def test_generate_dataset_name_many_symbols(self):
        """Test dataset name generation with many symbols."""
        symbols = ["SYM" + str(i) for i in range(10)]
        name = generate_dataset_name(symbols, "1h", "test")
        
        # Should indicate there are more symbols
        self.assertIn("+", name)
    
    def test_format_size(self):
        """Test size formatting."""
        self.assertEqual(format_size(500), "500.0 B")
        self.assertEqual(format_size(1024), "1.0 KB")
        self.assertEqual(format_size(1024 * 1024), "1.0 MB")
        self.assertEqual(format_size(1024 * 1024 * 1024), "1.0 GB")


if __name__ == "__main__":
    unittest.main()
