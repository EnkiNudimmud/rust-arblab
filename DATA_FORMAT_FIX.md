# Data Format Fix - Complete Solution

## Problem Summary

User reported: **"Unknown data format - cannot extract symbols"** despite having 304,422 rows of data loaded for 6 symbols (GDX, GDXJ, GLD, IAU, PSLV, SLV).

### Root Cause

The data persistence system was saving and loading data in **row-based MultiIndex** format `(timestamp, symbol)` on the index, but the UI expected **column-based MultiIndex** format with `(symbol, ohlcv)` on the columns. When data was loaded from saved datasets, it would come back in the wrong format, causing extraction to fail.

### Data Format Journey

1. **API Fetch** → Flat DataFrame with columns: `[timestamp, symbol, open, high, low, close, volume]`
2. **Data Loader Conversion** → Column-based MultiIndex with structure:
   ```
   Index: DatetimeIndex (timestamps)
   Columns: MultiIndex [(symbol, ohlcv), ...]
   Example: ('GLD', 'close'), ('GLD', 'open'), ('GDXJ', 'close'), ...
   ```
3. **Save to Disk** → Flattened back to row format and saved as Parquet
4. **Load from Disk** → **BUG HERE** - Restored as row-based MultiIndex `(timestamp, symbol)` on index ❌
5. **Lab Extraction** → Expected column-based MultiIndex, couldn't find symbols ❌

## Solution Implemented

### 1. Fixed `load_dataset()` in `data_persistence.py`

**Before:**
```python
if "timestamp" in df.columns and "symbol" in df.columns:
    df = df.set_index(["timestamp", "symbol"]).sort_index()  # Row-based MultiIndex ❌
```

**After:**
```python
if "timestamp" in df.columns and "symbol" in df.columns:
    df_indexed = df.set_index(["timestamp", "symbol"]).sort_index()
    # Unstack to get symbols as columns
    df = df_indexed.unstack(level='symbol')
    # Swap levels to (symbol, ohlcv)
    df.columns = df.columns.swaplevel(0, 1)
    df = df.sort_index(axis=1)  # Column-based MultiIndex ✅
```

### 2. Fixed `merge_dataframes()` in `data_persistence.py`

**Key Changes:**
- Detects whether input data is in column-based or row-based MultiIndex
- Converts both to flat format for merging
- **Always returns column-based MultiIndex format** to maintain consistency
- Handles all 3 modes: `append`, `update`, `replace`

**New Behavior:**
```python
# Input: Column MultiIndex (symbol, ohlcv)
# Output: Column MultiIndex (symbol, ohlcv) ✅

# Input: Row MultiIndex (timestamp, symbol)  
# Output: Column MultiIndex (symbol, ohlcv) ✅

# Input: Flat with symbol column
# Output: Column MultiIndex (symbol, ohlcv) ✅
```

### 3. Enhanced Detection in `lab_superspace_anomaly.py`

**Added comprehensive debugging:**
- Expander shows actual data structure (columns, index type, first 3 rows)
- Handles 5 different data formats:
  1. Dict format (already processed)
  2. Row-based MultiIndex → auto-convert
  3. Flat with symbol column → auto-convert
  4. Column-based MultiIndex → use directly ✅
  5. Unknown format → clear error with guidance

**Error Display:**
- Shows processing status per symbol
- Clear indicators: ✅ (300+ points), ⚠️ (<300 points), ❌ (missing)
- Stops execution with `st.stop()` if no data extracted
- Provides actionable next steps

## Testing Results

**Test Scenario:** 304,422 rows × 7 columns (6 symbols × 50,737 timestamps)

```
Step 1: Create flat data (304,422 rows)
Step 2: Convert to column MultiIndex (50,737 × 30 columns)
Step 3: Save to disk
Step 4: Load from disk
Result: ✅ Column MultiIndex (50,737 × 30)
Step 5: Extract per symbol
Result: ✅ All 6 symbols × 50,737 rows each
```

### What Changed After Fix

| Before | After |
|--------|-------|
| Loaded data: Row MultiIndex | Loaded data: Column MultiIndex |
| Extract: ❌ "Not found" | Extract: ✅ 50,737 rows |
| Format shown: `RangeIndex` | Format shown: `MultiIndex` with symbols |
| Error: "Unknown format" | Success: Auto-detection works |

## Files Modified

1. `/python/utils/data_persistence.py`
   - `load_dataset()` - Returns column-based MultiIndex
   - `merge_dataframes()` - Preserves column-based MultiIndex
   - Added `future_stack=True` to fix pandas warnings

2. `/app/pages/lab_superspace_anomaly.py`
   - Enhanced "Data Processing Details" expander (now expanded by default)
   - Shows actual columns, index type, first 3 rows
   - Better error messages with guidance
   - Stops execution cleanly if no data extracted

## How to Verify Fix

1. **Clear Streamlit cache:**
   - Go to http://localhost:8501
   - Click hamburger menu (☰) → "Clear cache"
   - Refresh page (Cmd+R)

2. **Check Data Loader:**
   - Load any dataset from saved datasets
   - Verify it loads as column-based MultiIndex

3. **Check Lab:**
   - Go to Superspace Analysis Lab → Analysis & Visualization
   - Expand "Data Processing Details"
   - Should show:
     - Columns type: `MultiIndex`
     - Symbols: [GDX, GDXJ, GLD, IAU, PSLV, SLV]
     - All symbols: ✅ with correct row counts

## User Action Required

**If issue persists:**
1. Clear browser cache and Streamlit cache
2. Re-load data from Data Loader page (this will save in new format)
3. Go back to Lab → Analysis tab
4. Check "Data Processing Details" expander for diagnostics

**Old datasets** saved before this fix will be automatically converted when loaded.

## Technical Details

### Column-based MultiIndex Structure
```python
Index: DatetimeIndex(['2024-01-01 00:00:00', ...], dtype='datetime64[ns]')
Columns: MultiIndex([
    ('GDX', 'close'), ('GDX', 'high'), ('GDX', 'low'), ('GDX', 'open'), ('GDX', 'volume'),
    ('GDXJ', 'close'), ('GDXJ', 'high'), ...
])
Shape: (50737, 30)  # timestamps × (6 symbols × 5 OHLCV)
```

### Extraction Example
```python
# Extract single symbol
gld_data = df['GLD']  # Returns DataFrame with columns: [close, high, low, open, volume]
# Shape: (50737, 5)

# Extract all selected symbols
data_dict = {}
for symbol in ['GDX', 'GDXJ', 'GLD', 'IAU', 'PSLV', 'SLV']:
    if symbol in df.columns.get_level_values(0):
        data_dict[symbol] = df[symbol].copy()
```

### Stack Operation for Merging
```python
# Column MultiIndex → Flat
df_flat = df.stack(level=0, future_stack=True).reset_index()
# Result: [timestamp, symbol, close, high, low, open, volume]

# Flat → Column MultiIndex
df_indexed = df_flat.set_index(['timestamp', 'symbol'])
df_columnar = df_indexed.unstack(level='symbol')
df_columnar.columns = df_columnar.columns.swaplevel(0, 1)
df_columnar = df_columnar.sort_index(axis=1)
```

## Prevention

This fix ensures:
- ✅ Data always loads in correct format
- ✅ Merging preserves correct format
- ✅ Lab auto-detects and converts if needed
- ✅ Clear error messages if format unrecognized
- ✅ User can see actual data structure for debugging
