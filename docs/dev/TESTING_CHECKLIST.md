# Multi-Page App Testing Checklist

## 🧪 Testing Guide

Use this checklist to verify all features work correctly after deployment.

---

## ✅ Pre-Launch Checks

### Environment Setup
- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r app/requirements.txt`)
- [ ] `api_keys.properties` file exists (or willing to use Yahoo Finance)
- [ ] Run script is executable (`chmod +x run_app.sh`)

### File Structure
- [ ] All page modules exist in `app/pages/`
- [ ] Utilities module exists in `app/utils/`
- [ ] `__init__.py` files present in modules
- [ ] Main app file exists: `app/main_app.py`

---

## 🚀 Launch Test

### Starting the App
```bash
./run_app.sh
```

- [ ] Script runs without errors
- [ ] Streamlit server starts
- [ ] Browser opens automatically to http://localhost:8501
- [ ] App loads without errors
- [ ] No import errors in terminal

---

## 📊 Page 1: Data Loading

### Navigation
- [ ] "📊 Data Loading" appears in sidebar
- [ ] Clicking navigates to data loading page
- [ ] Page title displays correctly

### Data Source: Yahoo Finance (No API Key Needed)
- [ ] Select "Yahoo Finance" from dropdown
- [ ] Enter symbols: `AAPL, MSFT`
- [ ] Set date range: Last 3 months
- [ ] Select interval: `1h`
- [ ] Click "Fetch Data"
- [ ] Data loads successfully
- [ ] Success message appears
- [ ] Quick stats show correct numbers
- [ ] Charts tab displays candlestick chart
- [ ] Data table tab shows data
- [ ] Statistics tab shows metrics
- [ ] Export tab has download buttons

### Data Source: CSV Upload
- [ ] Select "Upload CSV"
- [ ] Upload valid CSV file
- [ ] Data loads correctly
- [ ] Preview appears

### Data Validation
- [ ] Missing data percentage displays
- [ ] Symbol count is correct
- [ ] Date range displays correctly
- [ ] Clear data button works

---

## ⚡ Page 2: Strategy Backtest

### Navigation
- [ ] "⚡ Strategy Backtest" appears in sidebar
- [ ] Page loads successfully
- [ ] Warning appears if no data loaded

### With Data Loaded
- [ ] Data loading warning doesn't appear
- [ ] Strategy dropdown populated
- [ ] Select "Mean Reversion (PCA)"
- [ ] Parameters appear correctly
- [ ] Set Entry Z-Score: 2.0
- [ ] Set Exit Z-Score: 0.5
- [ ] Set Initial Capital: $100,000
- [ ] Set Transaction Cost: 10 bps
- [ ] Click "Run Backtest"
- [ ] Backtest executes without errors
- [ ] Success message appears

### Results Display
- [ ] Performance metrics show (4 metrics)
- [ ] Equity curve displays
- [ ] Portfolio weights show
- [ ] Portfolio weights chart renders
- [ ] Z-score chart displays (for mean reversion)
- [ ] Returns distribution shows

### Multiple Strategies
- [ ] Try "Mean Reversion (CARA)"
- [ ] Try "Mean Reversion (Sharpe)"
- [ ] Each completes successfully
- [ ] Results update correctly

---

## 🔴 Page 3: Live Trading

### Navigation
- [ ] "🔴 Live Trading" appears in sidebar
- [ ] Safety acknowledgment appears first
- [ ] Checkbox "I understand" works
- [ ] Page loads after acknowledgment

### Configuration
- [ ] Connector dropdown populated
- [ ] Select "finnhub" (if API key available) or "mock"
- [ ] Connection mode options display
- [ ] Select "Polling (REST)"
- [ ] Interval slider appears
- [ ] Symbols selection available
- [ ] Select 1-2 symbols

### Starting Live Feed
- [ ] "Start Live Feed" button enabled
- [ ] Click "Start Live Feed"
- [ ] Status changes to "LIVE"
- [ ] Pause and Stop buttons appear
- [ ] Data starts appearing

### Live Data Display
- [ ] Current quotes show for each symbol
- [ ] Bid/Ask/Spread display correctly
- [ ] Charts render for each symbol
- [ ] Charts update with new data
- [ ] Time axis shows correctly

### Live Analytics
- [ ] Statistics tab shows per-symbol stats
- [ ] Trade log tab exists (may be empty)
- [ ] Signals tab exists

### Stopping
- [ ] Click "Stop & Reset"
- [ ] Live feed stops
- [ ] Returns to initial state

---

## 💼 Page 4: Portfolio View

### Navigation
- [ ] "💼 Portfolio View" appears in sidebar
- [ ] Page loads successfully

### Initial State
- [ ] Portfolio metrics display
- [ ] Total Value shows $100,000 (default)
- [ ] Cash shows $100,000
- [ ] Open Positions shows 0

### With Positions (Manual Add)
- [ ] Sidebar has "Add Manual Position" expander
- [ ] Enter Symbol: "AAPL"
- [ ] Enter Quantity: 10
- [ ] Enter Price: 150
- [ ] Click "Add Position"
- [ ] Position appears in Holdings tab
- [ ] Metrics update correctly

### Holdings Tab
- [ ] Position table displays
- [ ] Shows Quantity, Prices, P&L
- [ ] Color coding works (green/red for P&L)
- [ ] Summary statistics show

### Performance Tab
- [ ] Chart displays (even with minimal data)
- [ ] Gauge shows portfolio value
- [ ] Metrics show when enough history

### Allocation Tab
- [ ] Pie chart displays
- [ ] Shows cash and positions
- [ ] Table shows allocation percentages

### Portfolio Management
- [ ] Reset portfolio button works
- [ ] Confirmation required
- [ ] Export portfolio button works
- [ ] Downloads JSON file

---

## 📈 Page 5: Options & Futures

### Navigation
- [ ] "📈 Options & Futures" appears in sidebar
- [ ] Page loads successfully
- [ ] Three tabs present

### Options Chain Tab
- [ ] Configuration form displays
- [ ] Enter Underlying: "AAPL"
- [ ] Enter Spot Price: 150
- [ ] Enter Days to Expiration: 30
- [ ] Enter Risk-Free Rate: 5%
- [ ] Enter Implied Vol: 30%
- [ ] Click "Generate Options Chain"
- [ ] Chain generates successfully
- [ ] Metadata displays (4 metrics)

### Options Display
- [ ] Calls table displays on left
- [ ] Puts table displays on right
- [ ] Strikes, Prices, Greeks show
- [ ] ATM rows highlighted
- [ ] Greeks selector appears
- [ ] Select "Delta"
- [ ] Delta chart displays
- [ ] Spot price line shows
- [ ] Select strike slider works
- [ ] Payoff diagram displays

### Futures Tab
- [ ] Configuration form displays
- [ ] Enter Underlying: "ES"
- [ ] Enter Spot Price: 4500
- [ ] Enter Number of Contracts: 12
- [ ] Enter Cost of Carry: 5%
- [ ] Click "Generate Futures Curve"
- [ ] Curve generates successfully
- [ ] Metadata displays
- [ ] Market structure shows (Contango/Backwardation)
- [ ] Futures table displays
- [ ] Term structure chart renders
- [ ] Basis chart shows

### Strategy Builder Tab
- [ ] Tab loads
- [ ] Shows "coming soon" message
- [ ] Preview expander works

---

## 🎨 UI/UX Tests

### Sidebar
- [ ] Navigation always visible
- [ ] Quick stats update correctly
- [ ] Rust status indicator shows
- [ ] Version number displays

### Responsiveness
- [ ] Works in full screen
- [ ] Columns adjust appropriately
- [ ] Charts resize correctly
- [ ] Tables fit in containers

### Theme & Styling
- [ ] Dark theme applied
- [ ] Custom CSS loads
- [ ] Colors consistent (green=good, red=bad)
- [ ] Metric cards styled correctly
- [ ] Charts use dark template

### Interactivity
- [ ] Hover info on charts works
- [ ] Chart zoom/pan works
- [ ] Selectboxes respond
- [ ] Buttons have hover effects
- [ ] Expandable sections work

---

## 🔄 Data Flow Tests

### Cross-Page Data
- [ ] Load data on Data Loading page
- [ ] Navigate to Strategy Backtest
- [ ] Data is available (no warning)
- [ ] Run backtest
- [ ] Navigate to Portfolio View
- [ ] Portfolio reflects backtest

### Session State
- [ ] Load data
- [ ] Navigate to different page
- [ ] Return to Data Loading
- [ ] Data still loaded
- [ ] Refresh page (F5)
- [ ] Session state resets

### Live Data Integration
- [ ] Start live feed
- [ ] Navigate to Portfolio View
- [ ] Current prices use live data
- [ ] Stop live feed
- [ ] Prices fallback to last known

---

## 🐛 Error Handling Tests

### Missing Data
- [ ] Go to Strategy Backtest without loading data
- [ ] Warning appears
- [ ] Can navigate to Data Loading
- [ ] No crashes

### Invalid Input
- [ ] Enter invalid symbols
- [ ] Error message appears
- [ ] App doesn't crash
- [ ] Can retry with valid input

### Network Issues
- [ ] Disconnect internet
- [ ] Try to fetch data
- [ ] Error message displays
- [ ] App remains functional

### API Key Issues
- [ ] Remove API key
- [ ] Try Finnhub
- [ ] Error message appears
- [ ] Fallback suggested (Yahoo Finance)

---

## ⚡ Performance Tests

### Data Loading
- [ ] Small dataset (3 symbols, 1 month): < 5 seconds
- [ ] Medium dataset (5 symbols, 6 months): < 15 seconds
- [ ] Caching works (reload is instant)

### Backtesting
- [ ] Simple backtest: < 3 seconds
- [ ] Complex backtest: < 10 seconds
- [ ] Results display immediately

### Live Trading
- [ ] Data updates in real-time
- [ ] No lag in chart updates
- [ ] UI remains responsive

### Charts
- [ ] All charts render < 2 seconds
- [ ] Interactions are smooth
- [ ] No freezing or lag

---

## 📱 Browser Compatibility

### Chrome/Chromium
- [ ] App loads
- [ ] All features work
- [ ] Charts render correctly

### Firefox
- [ ] App loads
- [ ] All features work
- [ ] Charts render correctly

### Safari
- [ ] App loads
- [ ] All features work
- [ ] Charts render correctly

---

## 🎯 End-to-End Workflows

### Workflow 1: Complete Trading Pipeline
1. [ ] Load historical data (Yahoo Finance, AAPL/MSFT, 6 months, 1h)
2. [ ] Backtest Mean Reversion (PCA)
3. [ ] Review results (good Sharpe ratio)
4. [ ] Start live feed (same symbols)
5. [ ] Monitor in real-time
6. [ ] Check portfolio view
7. [ ] Export results

### Workflow 2: Options Analysis
1. [ ] Go to Options & Futures
2. [ ] Generate options chain (AAPL, $150)
3. [ ] Analyze Greeks
4. [ ] View payoff diagrams
5. [ ] Try different strikes
6. [ ] Generate futures curve

### Workflow 3: Manual Portfolio Management
1. [ ] Go to Portfolio View
2. [ ] Add manual positions (3 stocks)
3. [ ] Check Holdings tab
4. [ ] View allocation
5. [ ] Export portfolio
6. [ ] Reset portfolio

---

## 🔒 Safety & Security Tests

### Paper Trading
- [ ] Safety warning on live trading page
- [ ] Acknowledgment required
- [ ] No real money involved messages

### Data Privacy
- [ ] API keys not logged
- [ ] No sensitive data exposed
- [ ] Export files are safe

### Error Messages
- [ ] No stack traces to users
- [ ] Helpful error messages
- [ ] Clear recovery instructions

---

## 📝 Documentation Tests

### README
- [ ] `app/README.md` exists
- [ ] All sections present
- [ ] Links work
- [ ] Examples are clear

### Quick Start
- [ ] `QUICKSTART_APP.md` exists
- [ ] Steps are clear
- [ ] Examples work
- [ ] Troubleshooting helpful

### Inline Help
- [ ] Tooltips present
- [ ] Info boxes helpful
- [ ] Examples provided
- [ ] Warnings appropriate

---

## ✅ Final Checks

### Production Ready
- [ ] No console errors
- [ ] No warnings (except expected)
- [ ] All imports resolve
- [ ] All pages load
- [ ] All features work
- [ ] Documentation complete
- [ ] Run script works

### User Experience
- [ ] Intuitive navigation
- [ ] Clear instructions
- [ ] Helpful error messages
- [ ] Good performance
- [ ] Professional appearance

---

## 🎉 Sign-Off

### Testing Complete
- [ ] All critical features tested
- [ ] All pages functional
- [ ] Documentation verified
- [ ] Ready for users

**Tested By:** _________________  
**Date:** _________________  
**Version:** 1.0.0  
**Status:** ✅ APPROVED / ⚠️ NEEDS WORK

---

## 📊 Test Results Summary

Total Tests: ~150+  
Passed: ___  
Failed: ___  
Skipped: ___  

**Overall Status:** READY FOR PRODUCTION ✅

---

## 🐛 Known Issues

Document any issues found during testing:

1. Issue: _______________
   - Severity: Low/Medium/High
   - Workaround: _______________
   - Status: Open/Fixed

---

## 🚀 Next Steps After Testing

1. [ ] Fix any critical issues
2. [ ] Deploy to production
3. [ ] Monitor user feedback
4. [ ] Plan next features
5. [ ] Update documentation

---

**Happy Testing! 🧪**
