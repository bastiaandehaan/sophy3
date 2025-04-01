# Sophy3 - Vectorized Trading System Summary

Generated on: 2025-04-01 15:53:54

## Core Components

## Recent Changes
- TradingProjectAnalyzer.py (modified on 2025-04-01)
- __init__.py (modified on 2025-04-01)
- config\__init__.py (modified on 2025-04-01)
- data\sources.py (modified on 2025-04-01)
- data\__init__.py (modified on 2025-04-01)

## Asset Class Support
- **Forex**: EMA periods=5/10/20, RSI=5, risk=1.0%
- **Crypto**: EMA periods=5/10/20, RSI=5, risk=1.0%, volatility factor=1.5

## Supported Trading Instruments
- **Forex**: EURUSD, GBPUSD
- **Crypto**: BTCUSD
- **Stocks**: TSLA, NFLX, META, GOOGL, MSFT, AMZN, AAPL
- **Indices**: FTSE, NDX, SPX, NKY, DAX, DJI

## Vectorized Implementation Details
**Indicators Used:**
- ATR
- EMA
- MovingAverage
- RSI
- StdDev

**Libraries:**
- matplotlib
- numpy
- pandas

**MT5 Integration:**
- initialize
- order_send
- order_send

**Optimization Features:**
- optimize
- optimize
- walk_forward
- monte_carlo
- parameter_opt
- optimize
- optimize
- optimize

## Configuration Files
- config.json

## Usage Examples
1. Example command:
   ```
   python scripts/backtest.py --symbol EURUSD --timeframe H1 --capital 10000 --risk 0.01 --detailed
   ```
2. Example command:
   ```
   python scripts/live_trade.py --account 123456 --password xxx --server MyBroker --symbols EURUSD GBPUSD --timeframe H1
   ```
