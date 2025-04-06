#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sophy3 Backtesting tool
Verantwoordelijk voor het uitvoeren van backtests op historische data
"""

import argparse
from datetime import datetime
import json
import os
import sys
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict, List, Optional, Tuple, Union

# Voeg projectmap toe aan sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from src.data.sources import DataSourceFactory, HistoricalDataSource
from src.strategies.ema_strategy import multi_layer_ema_strategy
from src.strategies.params import get_default_params
from src.utils.time_utils import parse_timeframe
from config.config import get_config
import logging

# Configureer logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Onderdruk onnodige vectorbt waarschuwingen
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Sophy3 Backtesting Tool")

    # Verplichte parameters
    parser.add_argument("--symbol", type=str, help="Trading symbol (bijv. EURUSD)")
    parser.add_argument("--timeframe", type=str, help="Timeframe (bijv. H1, M15, D1)")
    parser.add_argument("--capital", type=float, default=100000.0,
                        help="Startkapitaal voor backtest")

    # Datumbereik
    parser.add_argument("--start-date", type=str,
                        help="Startdatum (format: YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="Einddatum (format: YYYY-MM-DD)")

    # Strategie parameters
    parser.add_argument("--ema-periods", type=str,
                        help="EMA periodes (bijv. '9,21,50')")
    parser.add_argument("--rsi-period", type=int, help="RSI periode")
    parser.add_argument("--rsi-overbought", type=int, help="RSI overbought niveau")
    parser.add_argument("--rsi-oversold", type=int, help="RSI oversold niveau")
    parser.add_argument("--volatility-factor", type=float,
                        help="Volatiliteitsfactor voor positionering")

    # Stop Loss / Take Profit parameters
    parser.add_argument("--sl-pct", type=float, default=0.02,
                        help="Stop loss percentage (bijv. 0.02 voor 2%)")
    parser.add_argument("--tp-pct", type=float, default=0.04,
                        help="Take profit percentage (bijv. 0.04 voor 4%)")
    parser.add_argument("--sl-atr", type=float,
                        help="Stop loss als ATR multiplier (overschrijft sl-pct)")
    parser.add_argument("--tp-atr", type=float,
                        help="Take profit als ATR multiplier (overschrijft tp-pct)")
    parser.add_argument("--atr-period", type=int, default=14,
                        help="Periode voor ATR berekening")

    # Optimalisatie parameters
    parser.add_argument("--optimize", action="store_true",
                        help="Voer parameter optimalisatie uit")
    parser.add_argument("--optimize-metric", type=str, default="sharpe",
                        choices=["sharpe", "sortino", "calmar", "total_return"],
                        help="Metric voor optimalisatie")

    # Uitvoer opties
    parser.add_argument("--detailed", action="store_true",
                        help="Toon gedetailleerde statistieken")
    parser.add_argument("--plot", action="store_true",
                        help="Toon plots van backtest resultaten")
    parser.add_argument("--save", action="store_true",
                        help="Sla backtest resultaten op")
    parser.add_argument("--save-dir", type=str, default="backtests",
                        help="Map om resultaten op te slaan")
    parser.add_argument("--data-source", type=str, default="mt5",
                        choices=["mt5", "csv", "parquet"], help="Bron voor historische data")

    return parser.parse_args()


def run_backtest(data: pd.DataFrame, params: dict, capital: float = 100000.0,
                 detailed: bool = False) -> Tuple[Dict, vbt.Portfolio]:
    """
    Voer een backtest uit met de gegeven parameters en historische data.

    Args:
        data: DataFrame met historische OHLCV data
        params: Strategie parameters
        capital: Startkapitaal voor de backtest
        detailed: Of gedetailleerde statistieken getoond moeten worden

    Returns:
        Tuple met backtest resultaten en portfolio object
    """
    # Haal strategie parameters uit
    ema_periods = params.get("ema_periods", [9, 21, 50])
    rsi_period = params.get("rsi_period", 14)
    rsi_oversold = params.get("rsi_oversold", 30)
    rsi_overbought = params.get("rsi_overbought", 70)

    # Stop Loss / Take Profit parameters
    sl_pct = params.get("sl_pct", 0.02)  # Default 2%
    tp_pct = params.get("tp_pct", 0.04)  # Default 4%

    # ATR-gebaseerde SL/TP indien gespecificeerd
    sl_atr = params.get("sl_atr", None)
    tp_atr = params.get("tp_atr", None)
    atr_period = params.get("atr_period", 14)

    # Genereer trading signalen
    entries, exits = multi_layer_ema_strategy(data, ema_periods=ema_periods,
                                              rsi_period=rsi_period, rsi_oversold=rsi_oversold, rsi_overbought=rsi_overbought)

    # Bereken ATR indien nodig voor dynamische SL/TP
    if sl_atr is not None or tp_atr is not None:
        data['atr'] = vbt.indicators.ATR.run(data['high'], data['low'], data['close'],
                                             window=atr_period).atr.to_numpy()

        # Converteer ATR multipliers naar percentage SL/TP
        if sl_atr is not None:
            # Dynamische SL gebaseerd op ATR en huidige prijs
            sl_pct_array = (data['atr'] * sl_atr) / data['close']
        else:
            sl_pct_array = np.full(len(data), sl_pct)

        if tp_atr is not None:
            # Dynamische TP gebaseerd op ATR en huidige prijs
            tp_pct_array = (data['atr'] * tp_atr) / data['close']
        else:
            tp_pct_array = np.full(len(data), tp_pct)
    else:
        # Vaste SL/TP percentages
        sl_pct_array = np.full(len(data), sl_pct)
        tp_pct_array = np.full(len(data), tp_pct)

    # Bereken portfolio performance met vectorbt
    portfolio = vbt.Portfolio.from_signals(close=data['close'], entries=entries,
                                           exits=exits, init_cash=capital, fees=0.0001,  # 1 pip trading kosten
                                           sl_stop=sl_pct_array, tp_stop=tp_pct_array, freq=parse_timeframe(data))

    # Haal backtest statistieken op
    stats = portfolio.stats()

    # Bereid resultaten voor
    results = {"Total Return": f"{stats['Total Return [%]']:.2f}%",
               "Sharpe Ratio": f"{stats['Sharpe Ratio']:.2f}",
               "Sortino Ratio": f"{stats['Sortino Ratio']:.2f}",
               "Calmar Ratio": f"{stats['Calmar Ratio']:.2f}",
               "Max Drawdown": f"{stats['Max Drawdown [%]']:.2f}%",
               "Win Rate": f"{stats['Win Rate [%]']:.2f}%",
               "Profit Factor": f"{stats['Profit Factor']:.2f}",
               "Total Trades": f"{stats['Total Trades']:.2f}", }

    # Toon resultaten
    print("\n" + "=" * 50)
    print(
        f"BACKTEST RESULTS FOR {data.index.name.upper() if data.index.name else 'UNKNOWN'} ({data.columns.name.upper() if data.columns.name else 'UNKNOWN'})")

    print("=" * 50)
    for key, value in results.items():
        print(f"{key}: {value}")

    if detailed:
        print("\n" + "=" * 50)
        print("DETAILED STATISTICS")
        print("=" * 50)
        print(stats)

    return results, portfolio


def optimize_parameters(data: pd.DataFrame, param_grid: Dict, capital: float = 100000.0,
                        metric: str = "sharpe") -> Tuple[Dict, Dict]:
    """
    Optimaliseer strategie parameters op basis van historische data.

    Args:
        data: DataFrame met historische OHLCV data
        param_grid: Grid van parameters om te optimaliseren
        capital: Startkapitaal voor de backtest
        metric: Metric om te optimaliseren (sharpe, sortino, calmar, total_return)

    Returns:
        Tuple met beste parameters en resultaten
    """
    logger.info(
        f"Starting parameter optimization for {data.index.name} on {data.columns.name}")

    # Bereken alle mogelijke parameter combinaties
    param_combinations = []

    # EMA periodes
    ema_periods_options = param_grid.get("ema_periods", [[9, 21, 50]])

    # RSI parameters
    rsi_period_options = param_grid.get("rsi_period", [14])
    rsi_oversold_options = param_grid.get("rsi_oversold", [30])
    rsi_overbought_options = param_grid.get("rsi_overbought", [70])

    # SL/TP parameters
    sl_pct_options = param_grid.get("sl_pct", [0.02])
    tp_pct_options = param_grid.get("tp_pct", [0.04])
    sl_atr_options = param_grid.get("sl_atr", [None])
    tp_atr_options = param_grid.get("tp_atr", [None])
    atr_period_options = param_grid.get("atr_period", [14])

    # Genereer alle combinaties
    total_combinations = (len(ema_periods_options) * len(rsi_period_options) * len(
        rsi_oversold_options) * len(rsi_overbought_options) * len(sl_pct_options) * len(
        tp_pct_options) * len(sl_atr_options) * len(tp_atr_options) * len(
        atr_period_options))

    logger.info(f"Testing {total_combinations} parameter combinations")

    best_score = -float('inf') if metric != "max_drawdown" else float('inf')
    best_params = None
    best_results = None

    count = 0
    start_time = time.time()

    # Test elke parameter combinatie
    for ema_periods in ema_periods_options:
        for rsi_period in rsi_period_options:
            for rsi_oversold in rsi_oversold_options:
                for rsi_overbought in rsi_overbought_options:
                    for sl_pct in sl_pct_options:
                        for tp_pct in tp_pct_options:
                            for sl_atr in sl_atr_options:
                                for tp_atr in tp_atr_options:
                                    for atr_period in atr_period_options:
                                        # Sla over als zowel vaste als ATR-gebaseerde SL/TP zijn ingesteld
                                        if sl_atr is not None and tp_atr is not None and sl_pct != 0.02 and tp_pct != 0.04:
                                            continue

                                        # Stel parameters in
                                        params = {"ema_periods": ema_periods,
                                                  "rsi_period": rsi_period,
                                                  "rsi_oversold": rsi_oversold,
                                                  "rsi_overbought": rsi_overbought,
                                                  "sl_pct": sl_pct, "tp_pct": tp_pct,
                                                  "sl_atr": sl_atr, "tp_atr": tp_atr,
                                                  "atr_period": atr_period}

                                        # Voer backtest uit
                                        try:
                                            results, portfolio = run_backtest(data=data,
                                                                              params=params, capital=capital,
                                                                              detailed=False)

                                            # Converteer resultaatstring naar float
                                            if metric == "sharpe":
                                                score = float(
                                                    portfolio.stats()["Sharpe Ratio"])
                                            elif metric == "sortino":
                                                score = float(
                                                    portfolio.stats()["Sortino Ratio"])
                                            elif metric == "calmar":
                                                score = float(
                                                    portfolio.stats()["Calmar Ratio"])
                                            elif metric == "total_return":
                                                score = float(portfolio.stats()[
                                                                  "Total Return [%]"])
                                            elif metric == "max_drawdown":
                                                score = float(portfolio.stats()[
                                                                  "Max Drawdown [%]"])
                                                # Voor drawdown willen we minimaliseren
                                                score = -score

                                            # Controleer of dit de beste score is
                                            if ((
                                                    metric != "max_drawdown" and score > best_score) or (
                                                    metric == "max_drawdown" and score < best_score)):
                                                best_score = score
                                                best_params = params
                                                best_results = results

                                        except Exception as e:
                                            logger.warning(
                                                f"Error during optimization: {e}")

                                        count += 1
                                        if count % 10 == 0:
                                            elapsed = time.time() - start_time
                                            remaining = (elapsed / count) * (
                                                    total_combinations - count)
                                            logger.info(
                                                f"Progress: {count}/{total_combinations} combinations ({count / total_combinations * 100:.1f}%) - ETA: {remaining / 60:.1f} min")

    logger.info(f"Parameter optimization completed. Best {metric}: {best_score}")
    logger.info(f"Best parameters: {best_params}")

    return best_params, best_results


def plot_backtest_results(portfolio, symbol, timeframe):
    """
    Plot backtest resultaten met vectorbt.

    Args:
        portfolio: vectorbt Portfolio object
        symbol: Trading symbol
        timeframe: Timeframe
    """
    plt.figure(figsize=(14, 8))

    # Plot cumulatieve returns
    plt.subplot(2, 1, 1)
    portfolio.plot_cum_returns()
    plt.title(f"{symbol} {timeframe} - Cumulative Returns")
    plt.grid(True)

    # Plot drawdowns
    plt.subplot(2, 1, 2)
    portfolio.plot_drawdowns()
    plt.title(f"{symbol} {timeframe} - Drawdowns")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Plot trades
    portfolio.plot_trades()
    plt.title(f"{symbol} {timeframe} - Trades")
    plt.tight_layout()
    plt.show()


def save_results(results: Dict, portfolio, symbol: str, timeframe: str, params: Dict,
                 save_dir: str = "backtests"):
    """
    Sla backtest resultaten op.

    Args:
        results: Dictionary met backtest resultaten
        portfolio: vectorbt Portfolio object
        symbol: Trading symbol
        timeframe: Timeframe
        params: Strategie parameters
        save_dir: Map om resultaten op te slaan
    """
    # Maak map aan als deze niet bestaat
    os.makedirs(save_dir, exist_ok=True)

    # Genereer bestandsnaam
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{symbol}_{timeframe}_{timestamp}"

    # Bereid data voor
    output = {"symbol": symbol, "timeframe": timeframe, "timestamp": timestamp,
              "params": params, "results": results,
              "stats": {k: str(v) for k, v in portfolio.stats().items()}}

    # Sla JSON op
    with open(f"{save_dir}/{filename}.json", "w") as f:
        json.dump(output, f, indent=4)

    logger.info(f"Results saved to {save_dir}/{filename}.json")


def main():
    """Main functie voor backtest tool."""
    # Parse command line argumenten
    args = parse_args()

    # Print banner
    print("\n" + "=" * 80)
    print("Sophy3 - Backtesting Tool v4.2")
    print("=" * 80)

    start_time = time.time()

    # Bepaal datumbereik
    start_date = datetime.strptime(args.start_date,
                                   "%Y-%m-%d") if args.start_date else None
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else None

    if start_date and end_date:
        logger.info(
            f"Backtesting {args.symbol} on {args.timeframe} from {start_date} to {end_date}")
    else:
        logger.info(f"Backtesting {args.symbol} on {args.timeframe}")

    # Haal historische data op
    config = get_config()
    data_source = DataSourceFactory.create_source(args.data_source, config)

    data = data_source.get_historical_data(symbol=args.symbol, timeframe=args.timeframe,
                                           start_date=start_date, end_date=end_date)

    # Stel parameter waarden in
    params = get_default_params()

    # Update parameters van command line argumenten
    if args.ema_periods:
        params["ema_periods"] = [int(p) for p in args.ema_periods.split(",")]
    if args.rsi_period:
        params["rsi_period"] = args.rsi_period
    if args.rsi_oversold:
        params["rsi_oversold"] = args.rsi_oversold
    if args.rsi_overbought:
        params["rsi_overbought"] = args.rsi_overbought
    if args.volatility_factor:
        params["volatility_factor"] = args.volatility_factor

    # Stop Loss / Take Profit parameters
    params["sl_pct"] = args.sl_pct
    params["tp_pct"] = args.tp_pct
    if args.sl_atr:
        params["sl_atr"] = args.sl_atr
    if args.tp_atr:
        params["tp_atr"] = args.tp_atr
    params["atr_period"] = args.atr_period

    # Voer parameter optimalisatie uit indien gevraagd
    if args.optimize:
        # Definieer parameter grid voor optimalisatie
        param_grid = {
            "ema_periods": [[9, 21, 50], [5, 20, 50], [8, 21, 55], [10, 30, 90]],
            "rsi_period": [7, 14, 21], "rsi_oversold": [20, 30, 40],
            "rsi_overbought": [60, 70, 80], "sl_pct": [0.01, 0.02, 0.03],
            "tp_pct": [0.02, 0.04, 0.06], "sl_atr": [None, 1.0, 1.5, 2.0],
            "tp_atr": [None, 2.0, 3.0, 4.0], "atr_period": [14, 21]}

        best_params, _ = optimize_parameters(data=data, param_grid=param_grid,
                                             capital=args.capital, metric=args.optimize_metric)

        # Gebruik geoptimaliseerde parameters
        params = best_params

        logger.info(
            f"Starting backtest for {args.symbol} on {args.timeframe} with params: {params}")

    # Voer backtest uit
    results, portfolio = run_backtest(data=data, params=params, capital=args.capital,
                                      detailed=args.detailed)

    # Plot resultaten indien gevraagd
    if args.plot:
        plot_backtest_results(portfolio, args.symbol, args.timeframe)

    # Sla resultaten op indien gevraagd
    if args.save:
        save_results(results=results, portfolio=portfolio, symbol=args.symbol,
                     timeframe=args.timeframe, params=params, save_dir=args.save_dir)

    # Print uitvoeringstijd
    execution_time = time.time() - start_time
    print(f"\nBacktest completed in {execution_time:.2f} seconds")

    # Sluit data source
    if hasattr(data_source, 'close'):
        data_source.close()

    print(f"\nTotal execution time: {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()