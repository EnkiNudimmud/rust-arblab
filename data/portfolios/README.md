# Portfolio Storage

This directory contains saved portfolio configurations from the Portfolio Analytics Lab.

## Files

- `last_portfolio.json` - The most recently saved portfolio (automatically loaded on startup)
- `{portfolio_name}_{timestamp}.json` - Historical portfolio snapshots

## Structure

Each portfolio file contains:
- Portfolio name and creation timestamp
- Position details (symbols, quantities, prices, weights)
- Initial capital and cash balance
- Expected metrics (return, volatility, Sharpe ratio)
- Optional: Information Ratio, Sortino Ratio, Calmar Ratio

## Persistence

Portfolios are automatically saved to this directory when you:
1. Click "Save as Current Portfolio" in the Optimization tab
2. Click "Save Current Portfolio" in the Current Portfolio tab

The last saved portfolio is automatically loaded when you reopen the application.

## Docker Volume

When running in Docker, this directory is mounted as a volume to persist data across container restarts.
