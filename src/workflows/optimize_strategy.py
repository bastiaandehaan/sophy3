"""
Sophy3 - Strategy Parameter Optimizer
Functie: Optimaliseer strategie parameters via grid search voor maximale Sharpe ratio
Auteur: AI Trading Assistant
Laatste update: 2025-04-07

Gebruik:
  python src/workflows/optimize_strategy.py --symbol EURUSD --timeframe H1 --lookback 1000

Dependencies:
  - vectorbt
  - pandas
  - numpy
  - matplotlib
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import vectorbt as vbt

# Add project root to path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

# Import Sophy3 components
from src.data.sources import get_data
from src.strategies.ema_strategy import multi_layer_ema_strategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("optimize_strategy.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Sophy3 Strategy Parameter Optimizer")

    parser.add_argument("--symbol", type=str, default="EURUSD", help="Trading symbol")
    parser.add_argument("--timeframe", type=str, default="H1", help="Timeframe")
    parser.add_argument("--lookback", type=int, default=1000, help="Number of bars to use")
    parser.add_argument("--capital", type=float, default=10000, help="Initial capital")
    parser.add_argument("--save-dir", type=str, default="results", help="Directory to save results")

    return parser.parse_args()

def create_strategy_function(ema_short, ema_medium, ema_long, rsi_period,
                           rsi_oversold, rsi_overbought, volatility_factor):
    """
    Create a strategy function with fixed parameters for vectorbt grid search.

    Returns:
    --------
    function
        A function that takes price data and returns entry/exit signals
    """
    def strategy_func(price):
        # Create a DataFrame with OHLC data (assuming we only have close price)
        df = pd.DataFrame({
            'open': price,
            'high': price,
            'low': price,
            'close': price
        })

        # Generate signals using our strategy function
        entries, exits = multi_layer_ema_strategy(
            df,
            ema_periods=[ema_short, ema_medium, ema_long],
            rsi_period=rsi_period,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
            volatility_factor=volatility_factor
        )

        return entries, exits

    return strategy_func

def optimize_sl_tp(data, ema_short, ema_medium, ema_long, rsi_period,
                 rsi_oversold, rsi_overbought, volatility_factor, capital):
    """Optimize Stop Loss and Take Profit parameters."""
    logger.info("Optimizing Stop Loss and Take Profit parameters...")

    # Parameter ranges for SL/TP
    sl_atr_values = [1.0, 1.5, 2.0, 2.5]
    tp_atr_values = [2.0, 2.5, 3.0, 3.5]

    # Create a strategy function with fixed parameters
    strategy_func = create_strategy_function(
        ema_short, ema_medium, ema_long,
        rsi_period, rsi_oversold, rsi_overbought,
        volatility_factor
    )

    # Calculate ATR for dynamic SL/TP
    atr = vbt.indicators.ATR.run(
        data['high'], data['low'], data['close'], window=14
    ).atr.to_numpy()

    # Create grid of SL/TP values
    results = []
    best_sharpe = -np.inf
    best_params = None

    print("\nTesting Stop Loss and Take Profit combinations:")
    print("=" * 60)
    print(f"{'SL ATR':<8} {'TP ATR':<8} {'Sharpe':<8} {'Return %':<8} {'Drawdown %':<10} {'Trades':<6}")
    print("-" * 60)

    for sl_atr in sl_atr_values:
        for tp_atr in tp_atr_values:
            # Calculate SL/TP in percentage terms
            sl_pct = (atr * sl_atr) / data['close']
            tp_pct = (atr * tp_atr) / data['close']

            # Get entries and exits from strategy
            entries, exits = strategy_func(data['close'])

            # Run backtest with these SL/TP values
            pf = vbt.Portfolio.from_signals(
                data['close'],
                entries,
                exits,
                sl_stop=sl_pct,  # Dynamic SL based on ATR
                tp_stop=tp_pct,  # Dynamic TP based on ATR
                init_cash=capital,
                freq='H'
            )

            # Get key metrics
            sharpe = pf.sharpe_ratio()
            total_return = pf.total_return()
            max_dd = pf.max_drawdown()
            n_trades = pf.trades.count()

            # Store results
            result = {
                'sl_atr': sl_atr,
                'tp_atr': tp_atr,
                'sharpe_ratio': sharpe,
                'total_return': total_return,
                'max_drawdown': max_dd,
                'n_trades': n_trades
            }
            results.append(result)

            # Print current result
            print(f"{sl_atr:<8.1f} {tp_atr:<8.1f} {sharpe:<8.2f} {total_return*100:<8.2f} {max_dd*100:<10.2f} {n_trades:<6}")

            # Track best parameters
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = {
                    'sl_atr': sl_atr,
                    'tp_atr': tp_atr
                }

    print("\nBest SL/TP Parameters:")
    print(f"Stop Loss ATR Multiplier: {best_params['sl_atr']}")
    print(f"Take Profit ATR Multiplier: {best_params['tp_atr']}")
    print(f"Sharpe Ratio: {best_sharpe:.2f}")

    return best_params, pd.DataFrame(results)

def optimize_ema_periods(data, sl_atr, tp_atr, rsi_period,
                       rsi_oversold, rsi_overbought, volatility_factor, capital):
    """Optimize EMA period parameters."""
    logger.info("Optimizing EMA period parameters...")

    # Parameter ranges for EMAs
    ema_short_values = [8, 9, 10]
    ema_medium_values = [20, 21, 30]
    ema_long_values = [50, 100, 200]

    # Calculate ATR for dynamic SL/TP
    atr = vbt.indicators.ATR.run(
        data['high'], data['low'], data['close'], window=14
    ).atr.to_numpy()

    # Calculate SL/TP in percentage terms
    sl_pct = (atr * sl_atr) / data['close']
    tp_pct = (atr * tp_atr) / data['close']

    # Create grid of EMA values
    results = []
    best_sharpe = -np.inf
    best_params = None

    print("\nTesting EMA Period combinations:")
    print("=" * 75)
    print(f"{'Short EMA':<10} {'Medium EMA':<12} {'Long EMA':<10} {'Sharpe':<8} {'Return %':<8} {'Drawdown %':<10} {'Trades':<6}")
    print("-" * 75)

    for ema_short in ema_short_values:
        for ema_medium in ema_medium_values:
            for ema_long in ema_long_values:
                # Skip invalid combinations
                if not (ema_short < ema_medium < ema_long):
                    continue

                # Create a strategy function with these EMA parameters
                strategy_func = create_strategy_function(
                    ema_short, ema_medium, ema_long,
                    rsi_period, rsi_oversold, rsi_overbought,
                    volatility_factor
                )

                # Get entries and exits from strategy
                entries, exits = strategy_func(data['close'])

                # Run backtest with these parameters
                pf = vbt.Portfolio.from_signals(
                    data['close'],
                    entries,
                    exits,
                    sl_stop=sl_pct,
                    tp_stop=tp_pct,
                    init_cash=capital,
                    freq='H'
                )

                # Get key metrics
                sharpe = pf.sharpe_ratio()
                total_return = pf.total_return()
                max_dd = pf.max_drawdown()
                n_trades = pf.trades.count()

                # Store results
                result = {
                    'ema_short': ema_short,
                    'ema_medium': ema_medium,
                    'ema_long': ema_long,
                    'sharpe_ratio': sharpe,
                    'total_return': total_return,
                    'max_drawdown': max_dd,
                    'n_trades': n_trades
                }
                results.append(result)

                # Print current result
                print(f"{ema_short:<10} {ema_medium:<12} {ema_long:<10} {sharpe:<8.2f} {total_return*100:<8.2f} {max_dd*100:<10.2f} {n_trades:<6}")

                # Track best parameters
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = {
                        'ema_short': ema_short,
                        'ema_medium': ema_medium,
                        'ema_long': ema_long
                    }

    print("\nBest EMA Parameters:")
    print(f"Short EMA: {best_params['ema_short']}")
    print(f"Medium EMA: {best_params['ema_medium']}")
    print(f"Long EMA: {best_params['ema_long']}")
    print(f"Sharpe Ratio: {best_sharpe:.2f}")

    return best_params, pd.DataFrame(results)

def optimize_rsi_parameters(data, ema_short, ema_medium, ema_long,
                           sl_atr, tp_atr, volatility_factor, capital):
    """Optimize RSI parameters."""
    logger.info("Optimizing RSI parameters...")

    # Parameter ranges for RSI
    rsi_period_values = [14, 21]
    rsi_oversold_values = [30, 40]
    rsi_overbought_values = [60, 70]

    # Calculate ATR for dynamic SL/TP
    atr = vbt.indicators.ATR.run(
        data['high'], data['low'], data['close'], window=14
    ).atr.to_numpy()

    # Calculate SL/TP in percentage terms
    sl_pct = (atr * sl_atr) / data['close']
    tp_pct = (atr * tp_atr) / data['close']

    # Create grid of RSI values
    results = []
    best_sharpe = -np.inf
    best_params = None

    print("\nTesting RSI Parameter combinations:")
    print("=" * 75)
    print(f"{'RSI Period':<12} {'Oversold':<10} {'Overbought':<12} {'Sharpe':<8} {'Return %':<8} {'Drawdown %':<10} {'Trades':<6}")
    print("-" * 75)

    for rsi_period in rsi_period_values:
        for rsi_oversold in rsi_oversold_values:
            for rsi_overbought in rsi_overbought_values:
                # Skip invalid combinations
                if rsi_oversold >= rsi_overbought:
                    continue

                # Create a strategy function with these RSI parameters
                strategy_func = create_strategy_function(
                    ema_short, ema_medium, ema_long,
                    rsi_period, rsi_oversold, rsi_overbought,
                    volatility_factor
                )

                # Get entries and exits from strategy
                entries, exits = strategy_func(data['close'])

                # Run backtest with these parameters
                pf = vbt.Portfolio.from_signals(
                    data['close'],
                    entries,
                    exits,
                    sl_stop=sl_pct,
                    tp_stop=tp_pct,
                    init_cash=capital,
                    freq='H'
                )

                # Get key metrics
                sharpe = pf.sharpe_ratio()
                total_return = pf.total_return()
                max_dd = pf.max_drawdown()
                n_trades = pf.trades.count()

                # Store results
                result = {
                    'rsi_period': rsi_period,
                    'rsi_oversold': rsi_oversold,
                    'rsi_overbought': rsi_overbought,
                    'sharpe_ratio': sharpe,
                    'total_return': total_return,
                    'max_drawdown': max_dd,
                    'n_trades': n_trades
                }
                results.append(result)

                # Print current result
                print(f"{rsi_period:<12} {rsi_oversold:<10} {rsi_overbought:<12} {sharpe:<8.2f} {total_return*100:<8.2f} {max_dd*100:<10.2f} {n_trades:<6}")

                # Track best parameters
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = {
                        'rsi_period': rsi_period,
                        'rsi_oversold': rsi_oversold,
                        'rsi_overbought': rsi_overbought
                    }

    print("\nBest RSI Parameters:")
    print(f"RSI Period: {best_params['rsi_period']}")
    print(f"RSI Oversold: {best_params['rsi_oversold']}")
    print(f"RSI Overbought: {best_params['rsi_overbought']}")
    print(f"Sharpe Ratio: {best_sharpe:.2f}")

    return best_params, pd.DataFrame(results)

def optimize_volatility_factor(data, ema_short, ema_medium, ema_long,
                             rsi_period, rsi_oversold, rsi_overbought,
                             sl_atr, tp_atr, capital):
    """Optimize volatility factor parameter."""
    logger.info("Optimizing volatility factor parameter...")

    # Parameter range for volatility factor
    volatility_factor_values = [0.5, 0.75, 1.0, 1.25]

    # Calculate ATR for dynamic SL/TP
    atr = vbt.indicators.ATR.run(
        data['high'], data['low'], data['close'], window=14
    ).atr.to_numpy()

    # Calculate SL/TP in percentage terms
    sl_pct = (atr * sl_atr) / data['close']
    tp_pct = (atr * tp_atr) / data['close']

    # Create grid of volatility factor values
    results = []
    best_sharpe = -np.inf
    best_params = None

    print("\nTesting Volatility Factor values:")
    print("=" * 60)
    print(f"{'Vol Factor':<12} {'Sharpe':<8} {'Return %':<8} {'Drawdown %':<10} {'Trades':<6}")
    print("-" * 60)

    for volatility_factor in volatility_factor_values:
        # Create a strategy function with this volatility factor
        strategy_func = create_strategy_function(
            ema_short, ema_medium, ema_long,
            rsi_period, rsi_oversold, rsi_overbought,
            volatility_factor
        )

        # Get entries and exits from strategy
        entries, exits = strategy_func(data['close'])

        # Run backtest with these parameters
        pf = vbt.Portfolio.from_signals(
            data['close'],
            entries,
            exits,
            sl_stop=sl_pct,
            tp_stop=tp_pct,
            init_cash=capital,
            freq='H'
        )

        # Get key metrics
        sharpe = pf.sharpe_ratio()
        total_return = pf.total_return()
        max_dd = pf.max_drawdown()
        n_trades = pf.trades.count()

        # Store results
        result = {
            'volatility_factor': volatility_factor,
            'sharpe_ratio': sharpe,
            'total_return': total_return,
            'max_drawdown': max_dd,
            'n_trades': n_trades
        }
        results.append(result)

        # Print current result
        print(f"{volatility_factor:<12.2f} {sharpe:<8.2f} {total_return*100:<8.2f} {max_dd*100:<10.2f} {n_trades:<6}")

        # Track best parameters
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = {
                'volatility_factor': volatility_factor
            }

    print("\nBest Volatility Factor:")
    print(f"Volatility Factor: {best_params['volatility_factor']}")
    print(f"Sharpe Ratio: {best_sharpe:.2f}")

    return best_params, pd.DataFrame(results)

def final_backtest(data, optimal_params, capital):
    """Run a final backtest with the optimal parameters."""
    logger.info("Running final backtest with optimal parameters...")

    # Extract all parameters
    ema_short = optimal_params['ema_short']
    ema_medium = optimal_params['ema_medium']
    ema_long = optimal_params['ema_long']
    rsi_period = optimal_params['rsi_period']
    rsi_oversold = optimal_params['rsi_oversold']
    rsi_overbought = optimal_params['rsi_overbought']
    volatility_factor = optimal_params['volatility_factor']
    sl_atr = optimal_params['sl_atr']
    tp_atr = optimal_params['tp_atr']

    # Calculate ATR for dynamic SL/TP
    atr = vbt.indicators.ATR.run(
        data['high'], data['low'], data['close'], window=14
    ).atr.to_numpy()

    # Calculate SL/TP in percentage terms
    sl_pct = (atr * sl_atr) / data['close']
    tp_pct = (atr * tp_atr) / data['close']

    # Generate signals
    df = data.copy()
    entries, exits = multi_layer_ema_strategy(
        df,
        ema_periods=[ema_short, ema_medium, ema_long],
        rsi_period=rsi_period,
        rsi_oversold=rsi_oversold,
        rsi_overbought=rsi_overbought,
        volatility_factor=volatility_factor
    )

    # Run backtest
    portfolio = vbt.Portfolio.from_signals(
        data['close'],
        entries,
        exits,
        sl_stop=sl_pct,
        tp_stop=tp_pct,
        init_cash=capital,
        freq='H'
    )

    # Get detailed statistics
    stats = portfolio.stats()

    print("\nFinal Backtest Results with Optimal Parameters:")
    print("=" * 60)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # Plot results using matplotlib instead of relying on portfolio.equity()
    try:
        fig, axs = plt.subplots(3, 1, figsize=(12, 16), gridspec_kw={"height_ratios": [2, 1, 1]})

        # Plot price and trades
        axs[0].plot(data.index, data["close"], label="Close Price", color="blue", alpha=0.6)
        entry_points = entries[entries].index
        exit_points = exits[exits].index

        if len(entry_points) > 0:
            axs[0].scatter(entry_points, data.loc[entry_points, "close"], color="green", marker="^", s=100, label="Entries")
        if len(exit_points) > 0:
            axs[0].scatter(exit_points, data.loc[exit_points, "close"], color="red", marker="v", s=100, label="Exits")

        axs[0].set_title(f"Price and Trades")
        axs[0].legend()
        axs[0].grid(True)

        # Plot RSI
        if "rsi" in df.columns:
            axs[1].plot(df.index, df["rsi"], label="RSI", color="blue")
            axs[1].axhline(y=rsi_overbought, color="r", linestyle="--", alpha=0.3)
            axs[1].axhline(y=rsi_oversold, color="g", linestyle="--", alpha=0.3)
            axs[1].set_title("RSI Indicator")
            axs[1].legend()
            axs[1].grid(True)

        # Plot equity curve using portfolio value instead of equity attribute
        # This is the fix for the "Portfolio object has no attribute 'equity'" error
        cum_returns = portfolio.cum_returns()
        equity_curve = capital * (1 + cum_returns)
        axs[2].plot(equity_curve.index, equity_curve.values, label="Equity Curve")
        axs[2].set_title("Equity Curve")
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error(f"Error plotting results: {e}")
        print(f"Error plotting results: {e}")

    return stats, portfolio

def save_results(optimal_params, symbol, timeframe, save_dir="results"):
    """Save optimization results to a file."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{save_dir}/optimal_params_{symbol}_{timeframe}_{timestamp}.json"

    # Convert to regular Python types for JSON serialization
    save_params = {k: float(v) if isinstance(v, np.float64) else v
                  for k, v in optimal_params.items()}

    import json
    with open(filename, 'w') as f:
        json.dump(save_params, f, indent=4)

    logger.info(f"Saved optimal parameters to {filename}")
    print(f"Saved optimal parameters to {filename}")

def main():
    """Main function to run the optimization process."""
    start_time = time.time()
    args = parse_args()

    print("\n" + "=" * 80)
    print("Sophy3 - Strategy Parameter Optimizer v1.0")
    print("=" * 80)

    # Load data with much longer timeframe
    print(f"Loading data for {args.symbol} on {args.timeframe}...")

    # Set lookback to 5 years (1825 days) regardless of timeframe
    days_back = 1825  # 5 years of data

    # For higher timeframes, we might need even more
    if args.timeframe.startswith('D'):
        days_back = 1825  # 5 years
    elif args.timeframe.startswith('H'):
        days_back = 1825  # Also 5 years for hourly data
    elif args.timeframe.startswith('M'):
        # For minute data, we might use less to avoid memory issues
        if args.timeframe == 'M1':
            days_back = 365  # 1 year
        elif args.timeframe == 'M5':
            days_back = 730  # 2 years
        elif args.timeframe in ['M15', 'M30']:
            days_back = 1095  # 3 years
        else:
            days_back = 1825  # Default to 5 years
    else:
        days_back = 1825  # Default fallback to 5 years

    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(days_back))

    data = get_data(args.symbol, args.timeframe, start_date=start_date, end_date=end_date)

    if data is None or len(data) == 0:
        logger.error("Failed to load data")
        print("Error: Failed to load data. Please check your symbol and timeframe.")
        return

    print(f"Loaded {len(data)} bars of data from {data.index[0]} to {data.index[-1]}")

    # Starting parameters
    initial_params = {
        'ema_short': 9,
        'ema_medium': 21,
        'ema_long': 50,
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'volatility_factor': 0.5,
        'sl_atr': 1.5,
        'tp_atr': 3.0
    }

    # 1. Optimize SL/TP parameters first
    sl_tp_params, sl_tp_results = optimize_sl_tp(
        data,
        initial_params['ema_short'],
        initial_params['ema_medium'],
        initial_params['ema_long'],
        initial_params['rsi_period'],
        initial_params['rsi_oversold'],
        initial_params['rsi_overbought'],
        initial_params['volatility_factor'],
        args.capital
    )

    # 2. Optimize EMA periods
    ema_params, ema_results = optimize_ema_periods(
        data,
        sl_tp_params['sl_atr'],
        sl_tp_params['tp_atr'],
        initial_params['rsi_period'],
        initial_params['rsi_oversold'],
        initial_params['rsi_overbought'],
        initial_params['volatility_factor'],
        args.capital
    )

    # 3. Optimize RSI parameters
    rsi_params, rsi_results = optimize_rsi_parameters(
        data,
        ema_params['ema_short'],
        ema_params['ema_medium'],
        ema_params['ema_long'],
        sl_tp_params['sl_atr'],
        sl_tp_params['tp_atr'],
        initial_params['volatility_factor'],
        args.capital
    )

    # 4. Optimize volatility factor
    vol_params, vol_results = optimize_volatility_factor(
        data,
        ema_params['ema_short'],
        ema_params['ema_medium'],
        ema_params['ema_long'],
        rsi_params['rsi_period'],
        rsi_params['rsi_oversold'],
        rsi_params['rsi_overbought'],
        sl_tp_params['sl_atr'],
        sl_tp_params['tp_atr'],
        args.capital
    )

    # Combine all optimal parameters
    optimal_params = {**ema_params, **rsi_params, **vol_params, **sl_tp_params}

    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE - OPTIMAL PARAMETERS")
    print("=" * 80)
    print(f"Symbol: {args.symbol}, Timeframe: {args.timeframe}")
    print(f"EMA Periods: [{optimal_params['ema_short']}, {optimal_params['ema_medium']}, {optimal_params['ema_long']}]")
    print(f"RSI Period: {optimal_params['rsi_period']}")
    print(f"RSI Oversold/Overbought: {optimal_params['rsi_oversold']}/{optimal_params['rsi_overbought']}")
    print(f"Volatility Factor: {optimal_params['volatility_factor']}")
    print(f"Stop Loss ATR: {optimal_params['sl_atr']}")
    print(f"Take Profit ATR: {optimal_params['tp_atr']}")

    # Final backtest with all optimal parameters
    final_stats, final_portfolio = final_backtest(data, optimal_params, args.capital)

    # Save results
    save_results(optimal_params, args.symbol, args.timeframe, args.save_dir)

    elapsed_time = time.time() - start_time
    print(f"\nTotal optimization time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

if __name__ == "__main__":
    main()