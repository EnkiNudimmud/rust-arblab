# Data Recovery Guide

## Current Situation

Your Streamlit session has **corrupted data** in memory:
- Shape: (304,422, 5)
- Columns: [open, high, low, close, volume]
- **Missing**: timestamp and symbol information
- **Status**: Cannot be used for analysis ❌

## Why This Happened

The data lost its structure, likely due to:
1. Cached data from before the fix was applied
2. Incorrect data loading in an earlier session
3. Browser/Streamlit cache containing old format

## How to Fix (Choose One Method)

### Method 1: Clear and Reload (Fastest) ⚡

1. **In the Superspace Analysis Lab:**
   - Click the **"🗑️ Clear Corrupted Data"** button
   - Or click **"📂 Go to Data Loader"** button

2. **In Data Loader page:**
   - Go to **"🔄 Fetch Data"** tab
   - Enter your symbols: `GLD, SLV, PSLV` (or your preferred symbols)
   - Select date range (recommendation for 300+ points):
     - **1min**: 7 days (~10,000 points per symbol)
     - **1h**: 30 days (~720 points per symbol)
     - **1d**: 1 year (~365 points per symbol)
   - Click **"Fetch Data"**
   - Wait for success message

3. **Return to Superspace Analysis Lab:**
   - Navigate back to **Superspace Analysis Lab** → **Analysis & Visualization** tab
   - Check "Data Processing Details" expander
   - Should show: ✅ All symbols with correct counts

### Method 2: Clear Cache (Most Thorough) 🔄

1. **Clear Streamlit cache:**
   - Click hamburger menu (☰) in top right
   - Select **"Clear cache"**
   - Click **"Clear cache"** in the dialog

2. **Refresh browser:**
   - Press `Cmd+R` (Mac) or `Ctrl+R` (Windows/Linux)
   - Or click browser refresh button

3. **Start fresh:**
   - Go to **Data Loader** page
   - Follow steps 2-3 from Method 1 above

### Method 3: Check Saved Datasets 📂

If you had previously saved datasets:

1. **Go to Data Loader page**
2. **Click "📂 Saved Datasets" tab**
3. **Look for your datasets** (currently none exist)
4. **If found:** Click "📤 Load" button
5. **Return to Lab** and verify

**Note**: Currently you have **no saved datasets**. You'll need to fetch fresh data.

## Expected Results After Fix

### Data Processing Details Should Show:
```
✓ Type: MultiIndex
✓ Columns type: MultiIndex  
✓ Symbols: ['GLD', 'PSLV', 'SLV']
✓ Index type: DatetimeIndex
```

### Data Extraction Results Should Show:
```
✅ GLD: 10,000+ data points
✅ SLV: 10,000+ data points
✅ PSLV: 10,000+ data points
```

### Run Analysis Button:
- Should be **enabled** and ready to use
- No error messages
- Ready to run superspace analysis

## What Was Fixed

All the underlying code has been fixed:
- ✅ Data loading returns correct format (column-based MultiIndex)
- ✅ Data merging preserves format
- ✅ Symbol extraction works correctly
- ✅ Lab validation detects all data formats

**The fix is complete and tested.** You just need to get fresh data into your session.

## Recommended Fetch Settings

For optimal analysis (minimum 300 points per symbol):

| Frequency | Days | Expected Points | Use Case |
|-----------|------|-----------------|----------|
| **1min** | 7 | ~10,000 | High-frequency analysis |
| **1h** | 30 | ~720 | Medium-frequency analysis |
| **1d** | 365 | ~365 | Long-term analysis |

Choose based on your analysis needs. More data points = better statistical analysis.

## If Issues Persist

1. **Check browser console** for JavaScript errors (F12 → Console tab)
2. **Verify Streamlit is running** at http://localhost:8501
3. **Check terminal logs** for Python errors
4. **Try a different browser** (Chrome, Firefox, Safari)
5. **Restart Streamlit:**
   ```bash
   cd /Users/melvinalvarez/Documents/Workspace/rust-arblab
   pkill -f streamlit
   streamlit run app/Home.py --server.port 8501
   ```

## Testing the Fix

You can verify the fix is working by running this test:

```bash
cd /Users/melvinalvarez/Documents/Workspace/rust-arblab
python3 -c "
from python.utils.data_persistence import load_dataset, save_dataset
import pandas as pd
import numpy as np

# Create test data
data = []
for sym in ['TEST1', 'TEST2']:
    for i in range(1000):
        data.append({
            'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(minutes=i),
            'symbol': sym,
            'close': 100 + i * 0.1,
            'open': 100, 'high': 105, 'low': 95, 'volume': 1000
        })

df = pd.DataFrame(data)
save_dataset(df, 'test_fix', {})
loaded, _ = load_dataset('test_fix')

print(f'Loaded format: {type(loaded.columns).__name__}')
print(f'Is MultiIndex: {isinstance(loaded.columns, pd.MultiIndex)}')
if isinstance(loaded.columns, pd.MultiIndex):
    print(f'Symbols: {list(loaded.columns.get_level_values(0).unique())}')
    print('✅ Fix is working!')
else:
    print('❌ Issue detected')
"
```

If this prints `✅ Fix is working!`, then your system is ready.

## Summary

**Current Status:** Corrupted data in session (no timestamp/symbol info)  
**Fix Status:** ✅ Code is fixed and tested  
**Required Action:** Clear corrupted data and fetch fresh data  
**Time Needed:** 2-5 minutes  
**Expected Result:** Working analysis with all symbols properly extracted
