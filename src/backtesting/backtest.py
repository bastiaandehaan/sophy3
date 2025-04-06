"""
Sophy3 - Backtest Script
Functie: Vectorized backtesting met uitgebreide performance metrics
Auteur: AI Trading Assistant
Laatste update: 2025-04-07

Gebruik:
  python src/backtesting/backtest.py --symbol EURUSD --timeframe H1 --capital 10000 --detailed

Dependencies:
  - pandas
  - numpy
  - vectorbt
  - matplotlib
  - tqdm
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import vectorbt as vbt
from tqdm import tqdm

# Voeg de root directory toe aan sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Importeer Sophy3 componenten
from src.data.sources import get_data, initialize_mt5, shutdown_mt5
from src.strategies.ema_strategy import multi_layer_ema_strategy  # Geüpdatet
from src.strategies.params import get_strategy_params, get_risk_params
from src.utils.time_utils import detect_timeframe_frequency

# Logging setup
if not os.path.exists("output/logs"):
    os.makedirs("output/logs")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("output/logs/backtest.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Vectorbt instellingen
vbt.settings.set(
    portfolio={
        "fees": 0.0001,  # 0.01% fees per trade
    }
)



def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Sophy3 Backtesting Tool")

    # Data argumenten
    parser.add_argument("--symbol", type=str, default="EURUSD", help="Trading symbool")
    parser.add_argument("--timeframe", type=str, default="H1", help="Timeframe")
    parser.add_argument("--start-date", type=str, help="Start datum (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="Eind datum (YYYY-MM-DD)")
    parser.add_argument("--bars", type=int, default=1000, help="Aantal bars")
    parser.add_argument("--no-cache", action="store_true", help="Gebruik geen cache")
    parser.add_argument("--refresh-cache", action="store_true", help="Ververs cache")

    # MT5 argumenten
    parser.add_argument("--mt5-account", type=int, help="MT5 account nummer")
    parser.add_argument("--mt5-password", type=str, help="MT5 wachtwoord")
    parser.add_argument("--mt5-server", type=str, help="MT5 server")

    # Strategie argumenten
    parser.add_argument("--parameter-preset", type=str, choices=["forex", "crypto", "index", "stock"], help="Gebruik voorgedefinieerde parameters")
    parser.add_argument("--ema1", type=int, help="Korte EMA periode")
    parser.add_argument("--ema2", type=int, help="Middellange EMA periode")
    parser.add_argument("--ema3", type=int, help="Lange EMA periode")
    parser.add_argument("--rsi-period", type=int, help="RSI periode")
    parser.add_argument("--rsi-oversold", type=int, help="RSI oversold niveau")
    parser.add_argument("--rsi-overbought", type=int, help="RSI overbought niveau")
    parser.add_argument("--volatility-factor", type=float, help="Volatiliteit factor")

    # Risk parameters
    parser.add_argument("--capital", type=float, default=10000, help="Initieel kapitaal")
    parser.add_argument("--risk", type=float, default=0.01, help="Risico per trade")
    parser.add_argument("--max-daily-loss", type=float, default=0.05, help="Maximaal dagelijks verlies")
    parser.add_argument("--max-total-loss", type=float, default=0.10, help="Maximaal totaal verlies")

    # Output opties
    parser.add_argument("--detailed", action="store_true", help="Toon gedetailleerde statistieken")
    parser.add_argument("--plot", action="store_true", help="Toon grafieken")
    parser.add_argument("--save-results", action="store_true", help="Sla resultaten op")
    parser.add_argument("--output-dir", type=str, default="output/results", help="Output directory")

    # Optimalisatie
    parser.add_argument("--optimize", action="store_true", help="Voer parameteroptimalisatie uit")
    parser.add_argument("--optimize-param", type=str, choices=["ema", "rsi", "volatility"], help="Parameter om te optimaliseren")
    parser.add_argument("--optimize-metric", type=str, default="sharpe", choices=["sharpe", "sortino", "calmar"], help="Metric om te optimaliseren")

    return parser.parse_args()

def run_backtest(df, symbol, timeframe, strategy_params, risk_params, initial_capital=10000.0, detailed=False):
    """
    Voert een vectorized backtest uit op de gegeven data.

    Parameters:
    -----------
    df : pandas.DataFrame
        OHLCV data
    symbol : str
        Trading symbool
    timeframe : str
        Timeframe
    strategy_params : dict
        Strategie parameters
    risk_params : dict
        Risk management parameters
    initial_capital : float
        Initieel kapitaal
    detailed : bool
        Toon gedetailleerde statistieken

    Returns:
    --------
    dict, vbt.Portfolio
        Backtestresultaten en portfolio object
    """
    start_time = time.time()
    logger.info(f"Starting backtest for {symbol} on {timeframe} with params: {strategy_params}")

    # Genereer signalen
    entries, exits = multi_layer_ema_strategy(  # Geüpdatet
        df,
        ema_periods=strategy_params["ema_periods"],
        rsi_period=strategy_params["rsi_period"],
        rsi_oversold=strategy_params["rsi_oversold"],
        rsi_overbought=strategy_params["rsi_overbought"],
        volatility_factor=strategy_params["volatility_factor"],
    )

    # Detecteer frequentie
    freq = detect_timeframe_frequency(df.index)

    # Simuleer portfolio met vectorbt
    portfolio = vbt.Portfolio.from_signals(
        close=df['close'],
        entries=entries,
        exits=exits,
        init_cash=initial_capital,
        size=risk_params["risk_per_trade"] * initial_capital,  # Dynamische positiegrootte
        freq=freq,
    )

    # Bereken uitgebreide metrics
    metrics = {
        "total_return": portfolio.total_return(),
        "sharpe_ratio": portfolio.sharpe_ratio(),
        "sortino_ratio": portfolio.sortino_ratio(),
        "calmar_ratio": portfolio.calmar_ratio(),
        "max_drawdown": portfolio.max_drawdown(),
        "win_rate": portfolio.trades.win_rate() * 100,
        "profit_factor": portfolio.trades.profit_factor(),
        "total_trades": portfolio.trades.count(),
    }

    # Print basisresultaten
    print("\n" + "=" * 50)
    print(f"BACKTEST RESULTS FOR {symbol} ({timeframe})")
    print("=" * 50)
    for key, value in metrics.items():
        if "rate" in key or "return" in key or "drawdown" in key:
            print(f"{key.replace('_', ' ').title()}: {value:.2f}%")
        else:
            print(f"{key.replace('_', ' ').title()}: {value:.2f}")

    # Gedetailleerde statistieken
    if detailed:
        print("\n" + "=" * 50)
        print("DETAILED STATISTICS")
        print("=" * 50)
        stats = portfolio.stats()
        print(stats)

    elapsed_time = time.time() - start_time
    print(f"\nBacktest completed in {elapsed_time:.2f} seconds")
    return metrics, portfolio

def optimize_parameters(df, symbol, base_params, risk_params, param_to_optimize, optimization_metric, initial_capital=10000.0):
    """
    Voert parameteroptimalisatie uit met vectorbt.

    Parameters:
    -----------
    df : pandas.DataFrame
        OHLCV data
    symbol : str
        Trading symbool
    base_params : dict
        Basis strategie parameters
    risk_params : dict
        Risk management parameters
    param_to_optimize : str
        Parameter type om te optimaliseren ("ema", "rsi", "volatility")
    optimization_metric : str
        Metric om te optimaliseren
    initial_capital : float
        Initieel kapitaal

    Returns:
    --------
    dict, dict, pd.DataFrame
        Optimale parameters, beste resultaat, en alle resultaten
    """
    logger.info(f"Starting parameter optimization for {param_to_optimize} on {symbol}")
    start_time = time.time()

    optimization_results = []
    freq = detect_timeframe_frequency(df.index)

    if param_to_optimize == "ema":
        ema1_values = [5, 8, 10, 12, 15, 20]
        ema2_values = [20, 30, 40, 50, 60]
        ema3_values = [100, 150, 200, 250]

        total_combinations = sum(1 for e1 in ema1_values for e2 in ema2_values for e3 in ema3_values if e1 < e2 < e3)
        progress_bar = tqdm(total=total_combinations, desc="EMA Optimization")

        for ema1 in ema1_values:
            for ema2 in ema2_values:
                for ema3 in ema3_values:
                    if ema1 < ema2 < ema3:  # Geüpdatet: e2 -> ema2
                        test_params = base_params.copy()
                        test_params["ema_periods"] = [ema1, ema2, ema3]
                        entries, exits = multi_layer_ema_strategy(df, **test_params)  # Geüpdatet
                        portfolio = vbt.Portfolio.from_signals(df['close'], entries=entries, exits=exits, init_cash=initial_capital, freq=freq)
                        metrics = {
                            "ema1": ema1, "ema2": ema2, "ema3": ema3,
                            "sharpe_ratio": portfolio.sharpe_ratio(),
                            "sortino_ratio": portfolio.sortino_ratio(),
                            "calmar_ratio": portfolio.calmar_ratio(),
                            "total_return": portfolio.total_return(),
                            "max_drawdown": portfolio.max_drawdown(),
                        }
                        optimization_results.append(metrics)
                        progress_bar.update(1)
        progress_bar.close()

    elif param_to_optimize == "rsi":
        rsi_periods = [5, 8, 10, 14, 20]
        oversold_values = [20, 25, 30, 35, 40]
        overbought_values = [60, 65, 70, 75, 80]

        total_combinations = sum(1 for p in rsi_periods for os in oversold_values for ob in overbought_values if os < ob)
        progress_bar = tqdm(total=total_combinations, desc="RSI Optimization")

        for rsi_period in rsi_periods:
            for oversold in oversold_values:
                for overbought in overbought_values:
                    if oversold < overbought:
                        test_params = base_params.copy()
                        test_params.update({"rsi_period": rsi_period, "rsi_oversold": oversold, "rsi_overbought": overbought})
                        entries, exits = multi_layer_ema_strategy(df, **test_params)  # Geüpdatet
                        portfolio = vbt.Portfolio.from_signals(df['close'], entries=entries, exits=exits, init_cash=initial_capital, freq=freq)
                        metrics = {
                            "rsi_period": rsi_period, "rsi_oversold": oversold, "rsi_overbought": overbought,
                            "sharpe_ratio": portfolio.sharpe_ratio(),
                            "sortino_ratio": portfolio.sortino_ratio(),
                            "calmar_ratio": portfolio.calmar_ratio(),
                            "total_return": portfolio.total_return(),
                            "max_drawdown": portfolio.max_drawdown(),
                        }
                        optimization_results.append(metrics)
                        progress_bar.update(1)
        progress_bar.close()

    elif param_to_optimize == "volatility":
        volatility_factors = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        progress_bar = tqdm(total=len(volatility_factors), desc="Volatility Optimization")

        for vol_factor in volatility_factors:
            test_params = base_params.copy()
            test_params["volatility_factor"] = vol_factor
            entries, exits = multi_layer_ema_strategy(df, **test_params)  # Geüpdatet
            portfolio = vbt.Portfolio.from_signals(df['close'], entries=entries, exits=exits, init_cash=initial_capital, freq=freq)
            metrics = {
                "volatility_factor": vol_factor,
                "sharpe_ratio": portfolio.sharpe_ratio(),
                "sortino_ratio": portfolio.sortino_ratio(),
                "calmar_ratio": portfolio.calmar_ratio(),
                "total_return": portfolio.total_return(),
                "max_drawdown": portfolio.max_drawdown(),
            }
            optimization_results.append(metrics)
            progress_bar.update(1)
        progress_bar.close()

    # Converteer naar DataFrame
    results_df = pd.DataFrame(optimization_results)
    if results_df.empty:
        logger.error("No valid optimization results found")
        return None, None, None

    # Sorteer op optimalisatie metric
    results_df = results_df.sort_values(optimization_metric, ascending=False)
    best_params = results_df.iloc[0].to_dict()

    # Update optimale parameters
    optimal_params = base_params.copy()
    if param_to_optimize == "ema":
        optimal_params["ema_periods"] = [int(best_params["ema1"]), int(best_params["ema2"]), int(best_params["ema3"])]
    elif param_to_optimize == "rsi":
        optimal_params.update({
            "rsi_period": int(best_params["rsi_period"]),
            "rsi_oversold": int(best_params["rsi_oversold"]),
            "rsi_overbought": int(best_params["rsi_overbought"]),
        })
    elif param_to_optimize == "volatility":
        optimal_params["volatility_factor"] = float(best_params["volatility_factor"])

    elapsed_time = time.time() - start_time
    print(f"\nOptimization completed in {elapsed_time:.2f} seconds")
    print("Top 5 Results:")
    print(results_df.head(5).to_string())
    return optimal_params, best_params, results_df

def plot_backtest_results(df, portfolio, symbol, timeframe):
    """Plot backtest resultaten."""
    print(f"\nPlotting backtest results for {symbol} ({timeframe})...")
    fig, axs = plt.subplots(3, 1, figsize=(12, 16), gridspec_kw={"height_ratios": [2, 1, 1]})

    # Plot 1: Prijs en trades
    ax1 = axs[0]
    ax1.plot(df.index, df["close"], label="Close Price", color="blue", alpha=0.6)
    entries = portfolio.entries[portfolio.entries]
    exits = portfolio.exits[portfolio.exits]
    ax1.scatter(entries.index, df.loc[entries.index, "close"], color="green", marker="^", s=100, label="Entries")
    ax1.scatter(exits.index, df.loc[exits.index, "close"], color="red", marker="v", s=100, label="Exits")
    ax1.set_title(f"{symbol} {timeframe} - Price and Trades")
    ax1.legend()
    ax1.grid(True)

    # Plot 2: RSI (indien beschikbaar)
    ax2 = axs[1]
    if "rsi" in df.columns:
        ax2.plot(df.index, df["rsi"], label="RSI", color="blue")
        ax2.axhline(y=70, color="r", linestyle="--", alpha=0.3)
        ax2.axhline(y=30, color="g", linestyle="--", alpha=0.3)
        ax2.set_title("RSI Indicator")
        ax2.legend()
        ax2.grid(True)

    # Plot 3: Equity curve
    ax3 = axs[2]
    portfolio.plot_value(ax=ax3)
    ax3.set_title("Equity Curve")
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

def save_results(metrics, portfolio, symbol, timeframe, output_dir="output/results"):
    """Slaat backtest resultaten op."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"{symbol}_{timeframe}_{timestamp}_results.json")
    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=4, default=str)

    trades_file = os.path.join(output_dir, f"{symbol}_{timeframe}_{timestamp}_trades.csv")
    portfolio.trades.records.to_pandas().to_csv(trades_file, index=False)

    equity_file = os.path.join(output_dir, f"{symbol}_{timeframe}_{timestamp}_equity.csv")
    portfolio.value().to_csv(equity_file)

    logger.info(f"Backtest results saved to {output_dir}")

def main():
    """Hoofdfunctie voor backtesting."""
    overall_start_time = time.time()
    args = parse_args()

    print("\n" + "=" * 80)
    print("Sophy3 - Backtesting Tool v4.2")
    print("=" * 80)

    # Initialiseer MT5
    if args.mt5_account and args.mt5_password:
        if not initialize_mt5(args.mt5_account, args.mt5_password, args.mt5_server):
            logger.error("MT5 initialization failed")
            return

    # Bepaal start- en einddatums
    end_date = datetime.now() if args.end_date is None else datetime.strptime(args.end_date, "%Y-%m-%d")
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    else:
        days = args.bars
        if args.timeframe.startswith("M"):
            mins = int(args.timeframe[1:]) if len(args.timeframe) > 1 else 1
            days = int(args.bars * mins / (60 * 24)) + 1
        elif args.timeframe.startswith("H"):
            hours = int(args.timeframe[1:]) if len(args.timeframe) > 1 else 1
            days = int(args.bars * hours / 24) + 1
        elif args.timeframe.startswith("D"):
            days = args.bars
        start_date = end_date - timedelta(days=days * 2)

    logger.info(f"Backtesting {args.symbol} on {args.timeframe} from {start_date} to {end_date}")

    # Haal data op
    df = get_data(
        args.symbol,
        args.timeframe,
        start_date=start_date,
        end_date=end_date,
        use_cache=not args.no_cache,
        refresh_cache=args.refresh_cache,
    )
    if df is None or len(df) == 0:
        logger.error("No data received")
        return

    # Haal parameters op
    asset_class = args.parameter_preset if args.parameter_preset else args.symbol
    strategy_params = get_strategy_params(asset_class)
    risk_params = get_risk_params(asset_class)

    # Overschrijf parameters indien opgegeven
    if args.ema1 and args.ema2 and args.ema3:
        strategy_params["ema_periods"] = [args.ema1, args.ema2, args.ema3]
    if args.rsi_period:
        strategy_params["rsi_period"] = args.rsi_period
    if args.rsi_oversold:
        strategy_params["rsi_oversold"] = args.rsi_oversold
    if args.rsi_overbought:
        strategy_params["rsi_overbought"] = args.rsi_overbought
    if args.volatility_factor:
        strategy_params["volatility_factor"] = args.volatility_factor
    if args.risk:
        risk_params["risk_per_trade"] = args.risk
    if args.max_daily_loss:
        risk_params["max_daily_loss"] = args.max_daily_loss
    if args.max_total_loss:
        risk_params["max_total_loss"] = args.max_total_loss

    # Parameteroptimalisatie
    if args.optimize and args.optimize_param:
        strategy_params, best_params, results_df = optimize_parameters(
            df, args.symbol, strategy_params, risk_params, args.optimize_param, args.optimize_metric, args.capital
        )
        if strategy_params is None:
            logger.error("Parameter optimization failed")
            return
        if args.save_results and results_df is not None:
            output_dir = os.path.join(args.output_dir, "optimization")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_df.to_csv(os.path.join(output_dir, f"{args.symbol}_{args.timeframe}_{args.optimize_param}_{timestamp}.csv"))

    # Voer backtest uit
    metrics, portfolio = run_backtest(
        df, args.symbol, args.timeframe, strategy_params, risk_params, args.capital, args.detailed
    )
    if metrics is None or portfolio is None:
        logger.error("Backtest failed")
        return

    # Plot resultaten
    if args.plot:
        plot_backtest_results(df, portfolio, args.symbol, args.timeframe)

    # Sla resultaten op
    if args.save_results:
        save_results(metrics, portfolio, args.symbol, args.timeframe, args.output_dir)

    # Sluit MT5
    shutdown_mt5()

    overall_elapsed_time = time.time() - overall_start_time
    print(f"\nTotal execution time: {overall_elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()