import vectorbt as vbt
import MetaTrader5 as mt5
import schedule
import time
import argparse
import sys
import os
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategies.multi_layer_ema import multi_layer_ema_strategy
from strategies.params import get_params
from risk.manager import FTMORiskManager
from data.sources import get_historical_data

# Global variables
risk_manager = None
active_symbols = []
trading_active = True

def connect_mt5(account, password, server):
    """Connect to MT5 terminal"""
    if not mt5.initialize():
        print("MT5 initialization failed")
        return False

    # Login to account
    authorized = mt5.login(account, password=password, server=server)
    if not authorized:
        print(f"MT5 login failed: {mt5.last_error()}")
        mt5.shutdown()
        return False

    print(f"Connected to MT5: {mt5.account_info().server}")
    return True

def check_for_signals(symbol, timeframe):
    """Check for trading signals for a symbol"""
    global risk_manager

    if not trading_active:
        return

    try:
        # Get recent data (last 100 bars)
        data = get_historical_data(symbol, timeframe, lookback_periods=100)

        # Get parameters for this asset
        params = get_params(symbol)

        # Generate signals
        entries, exits = multi_layer_ema_strategy(data, **params)

        # Check most recent signal
        latest_idx = entries.index[-1]
        current_price = data.close.iloc[-1]

        # Entry signal
        if entries.iloc[-1] and not mt5.positions_get(symbol=symbol):
            # Calculate position size
            atr = vbt.ta.atr(data.high, data.low, data.close, 14).iloc[-1]
            stop_loss = current_price - (atr * 1.5)  # Stop loss at 1.5 ATR

            size = risk_manager.calculate_position_size(
                current_price,
                stop_loss,
                risk_per_trade=0.01
            )

            # Execute order via MT5
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": size,
                "type": mt5.ORDER_TYPE_BUY,
                "price": current_price,
                "sl": stop_loss,
                "tp": current_price + (atr * 2),  # Take profit at 2 ATR
                "comment": "sophy3_multi_ema",
                "type_time": mt5.ORDER_TIME_GTC,
            }

            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"Order failed: {result.comment}")
            else:
                print(f"Order placed: {symbol} BUY at {current_price}, size: {size}")

        # Exit signal
        elif exits.iloc[-1] and mt5.positions_get(symbol=symbol):
            positions = mt5.positions_get(symbol=symbol)
            for position in positions:
                # Close position
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": position.volume,
                    "type": mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,
                    "position": position.ticket,
                    "price": current_price,
                    "comment": "sophy3_multi_ema_exit",
                    "type_time": mt5.ORDER_TIME_GTC,
                }

                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    print(f"Exit order failed: {result.comment}")
                else:
                    print(f"Position closed: {symbol} at {current_price}")

    except Exception as e:
        print(f"Error checking signals for {symbol}: {e}")

def reset_daily_risk():
    """Reset daily risk limits (call at start of trading day)"""
    global risk_manager
    if risk_manager:
        risk_manager.reset_daily_pnl()
        print(f"Daily risk limits reset at {datetime.now()}")

def main():
    global risk_manager, active_symbols, trading_active

    parser = argparse.ArgumentParser(description='Run live trading with Multi-Layer EMA strategy')
    parser.add_argument('--account', type=int, required=True, help='MT5 account number')
    parser.add_argument('--password', type=str, required=True, help='MT5 password')
    parser.add_argument('--server', type=str, required=True, help='MT5 server')
    parser.add_argument('--symbols', type=str, nargs='+', required=True, help='Symbols to trade')
    parser.add_argument('--timeframe', type=str, default='H1', help='Timeframe (e.g. H1, D1)')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    args = parser.parse_args()

    # Connect to MT5
    if not connect_mt5(args.account, args.password, args.server):
        return

    # Initialize risk manager
    risk_manager = FTMORiskManager(initial_capital=args.capital)
    active_symbols = args.symbols

    # Schedule signal checks based on timeframe
    interval = "hour" if args.timeframe.lower() in ['h1', '1h'] else "day"

    for symbol in active_symbols:
        if interval == "hour":
            schedule.every().hour.at(":01").do(check_for_signals, symbol=symbol, timeframe=args.timeframe)
        else:
            schedule.every().day.at("00:01").do(check_for_signals, symbol=symbol, timeframe=args.timeframe)

    # Reset daily risk at beginning of trading day
    schedule.every().day.at("00:00").do(reset_daily_risk)

    print(f"Live trading started for {', '.join(active_symbols)} on {args.timeframe}")
    print("Press Ctrl+C to stop")

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        trading_active = False
        mt5.shutdown()

if __name__ == "__main__":
    main()
