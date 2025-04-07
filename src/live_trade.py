import argparse
import os
import sys
import time
from datetime import datetime

import MetaTrader5 as mt5
import schedule
import vectorbt as vbt

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.strategies.ema_strategy import simple_ema_strategy
from src.strategies.params import get_strategy_params
from src.risk import FTMORiskManager
from src.data.sources import get_data as get_historical_data  # Geüpdatet

# Global variables
risk_manager = None
active_symbols = []
trading_active = True

# Vooraf ingestelde configuratie (vul hier je echte gegevens in)
DEFAULT_ACCOUNT = 123456  # Vervang dit met je echte account nummer
DEFAULT_PASSWORD = "je_wachtwoord"  # Vervang dit met je echte wachtwoord
DEFAULT_SERVER = "je_server"  # Vervang dit met je echte server
DEFAULT_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY"]  # Vervang dit met je symbolen
DEFAULT_TIMEFRAME = "H1"
DEFAULT_CAPITAL = 10000


def connect_mt5(account=None, password=None, server=None):
    """Connect to MT5 terminal"""
    # Controleer eerst of MT5 al is geïnitialiseerd
    if mt5.terminal_info() is not None:
        print("MT5 is already initialized")
        return True

    if not mt5.initialize():
        print("MT5 initialization failed")
        return False

    # Login to account only if credentials are provided
    if account and password and server:
        authorized = mt5.login(account, password=password, server=server)
        if not authorized:
            print(f"MT5 login failed: {mt5.last_error()}")
            mt5.shutdown()
            return False
        print(f"Connected to MT5: {mt5.account_info().server}")
    else:
        # Check if we're already logged in
        if mt5.account_info() is None:
            print("Not logged in to MT5 and no credentials provided")
            return False
        print(f"Using existing MT5 connection: {mt5.account_info().server}")

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
        params = get_strategy_params(symbol)

        # Generate signals
        entries, exits, sl_pct, tp_pct = simple_ema_strategy(data, **params)

        # Check most recent signal
        latest_idx = entries.index[-1]
        current_price = data.close.iloc[-1]

        # Entry signal
        if entries.iloc[-1] and not mt5.positions_get(symbol=symbol):
            # Calculate position size
            stop_loss = current_price * (1 - sl_pct.iloc[-1])

            size = risk_manager.calculate_position_size(current_price, stop_loss,
                                                        risk_per_trade=0.01)

            # Execute order via MT5
            request = {"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol,
                       "volume": size, "type": mt5.ORDER_TYPE_BUY, "price": current_price,
                       "sl": stop_loss, "tp": current_price * (1 + tp_pct.iloc[-1]),
                       "comment": "sophy3_ema", "type_time": mt5.ORDER_TIME_GTC, }

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
                request = {"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol,
                           "volume": position.volume, "type": (
                        mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY),
                           "position": position.ticket, "price": current_price,
                           "comment": "sophy3_ema_exit", "type_time": mt5.ORDER_TIME_GTC, }

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

    parser = argparse.ArgumentParser(description="Run live trading with EMA strategy")
    parser.add_argument("--account", type=int, default=DEFAULT_ACCOUNT,
                        help="MT5 account number")
    parser.add_argument("--password", type=str, default=DEFAULT_PASSWORD,
                        help="MT5 password")
    parser.add_argument("--server", type=str, default=DEFAULT_SERVER, help="MT5 server")
    parser.add_argument("--symbols", type=str, nargs="+", default=DEFAULT_SYMBOLS,
                        help="Symbols to trade")
    parser.add_argument("--timeframe", type=str, default=DEFAULT_TIMEFRAME,
                        help="Timeframe (e.g. H1, D1)")
    parser.add_argument("--capital", type=float, default=DEFAULT_CAPITAL,
                        help="Initial capital")
    args = parser.parse_args()

    # Connect to MT5
    if not connect_mt5(args.account, args.password, args.server):
        return

    # Initialize risk manager
    risk_manager = FTMORiskManager(initial_capital=args.capital)
    active_symbols = args.symbols

    # Schedule signal checks based on timeframe
    interval = "hour" if args.timeframe.lower() in ["h1", "1h"] else "day"

    for symbol in active_symbols:
        if interval == "hour":
            schedule.every().hour.at(":01").do(check_for_signals, symbol=symbol,
                                               timeframe=args.timeframe)
        else:
            schedule.every().day.at("00:01").do(check_for_signals, symbol=symbol,
                                                timeframe=args.timeframe)

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