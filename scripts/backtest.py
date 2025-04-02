"""
Sophy3 - Backtest Script
Functie: Vectorized backtesting met performance metrics
Auteur: AI Trading Assistant
Laatste update: 2025-04-03

Gebruik:
  python scripts/backtest.py --symbol EURUSD --timeframe H1 --capital 10000 --risk 0.01 --detailed

Dependencies:
  - pandas
  - numpy
  - vectorbt
  - matplotlib
  - tqdm (voortgangsbalk)
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import json
import time
import vectorbt as vbt
from tqdm import tqdm
import gc

# Voeg de root directory toe aan sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importeer Sophy3 componenten
from strategies.multi_layer_ema import multi_layer_ema_strategy
from strategies.params import get_strategy_params, get_risk_params
from data.sources import get_data, initialize_mt5, shutdown_mt5
from risk.manager import FTMORiskManager

# Logging setup
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("backtest.log"), logging.StreamHandler()], )
logger = logging.getLogger(__name__)


def detect_timeframe_frequency(df_index):
    """
    Detecteert pandas frequentie string op basis van index.

    Parameters:
    -----------
    df_index : pandas.DatetimeIndex
        DatetimeIndex van het DataFrame

    Returns:
    --------
    str
        Pandas-compatibele frequentie string
    """
    if not isinstance(df_index, pd.DatetimeIndex) or len(df_index) < 2:
        return None

    # Bereken meest voorkomende interval (neem eerste 100 samples voor efficiëntie)
    sample_size = min(100, len(df_index) - 1)
    diffs = [(df_index[i + 1] - df_index[i]) for i in range(sample_size)]

    # Gebruik de meest voorkomende waarde
    from collections import Counter
    most_common_diff = Counter(diffs).most_common(1)[0][0]

    # Map naar pandas frequentie string
    seconds = most_common_diff.total_seconds()

    if seconds == 60:
        return "1min"
    elif seconds == 300:
        return "5min"
    elif seconds == 900:
        return "15min"
    elif seconds == 1800:
        return "30min"
    elif seconds == 3600:
        return "1H"
    elif seconds == 14400:
        return "4H"
    elif seconds == 86400:  # 1 dag
        return "1D"
    elif seconds == 604800:  # 7 dagen
        return "1W"

    # Fallback: retourneer een string op basis van seconden
    return f"{int(seconds)}S"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Sophy3 Backtesting Tool")

    # Data argumenten
    parser.add_argument("--symbol", type=str, default="EURUSD",
        help="Trading symbool (bijv. EURUSD)")
    parser.add_argument("--timeframe", type=str, default="H1",
        help="Timeframe (bijv. M15, H1, D1)")
    parser.add_argument("--start-date", type=str, help="Start datum (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="Eind datum (YYYY-MM-DD)")
    parser.add_argument("--bars", type=int, default=1000,
        help="Aantal bars voor backtesting")
    parser.add_argument("--no-cache", action="store_true",
        help="Data niet uit cache gebruiken")
    parser.add_argument("--refresh-cache", action="store_true",
        help="Forceer het verversen van de cache", )

    # MT5 argumenten
    parser.add_argument("--mt5-account", type=int, help="MT5 account nummer")
    parser.add_argument("--mt5-password", type=str, help="MT5 wachtwoord")
    parser.add_argument("--mt5-server", type=str, help="MT5 server")

    # Strategie argumenten
    parser.add_argument("--parameter-preset", type=str,
        choices=["forex", "crypto", "index", "stock"],
        help="Gebruik voorgedefinieerde parameters voor asset class", )
    parser.add_argument("--ema1", type=int, help="Korte EMA periode")
    parser.add_argument("--ema2", type=int, help="Middellange EMA periode")
    parser.add_argument("--ema3", type=int, help="Lange EMA periode")
    parser.add_argument("--rsi-period", type=int, help="RSI periode")
    parser.add_argument("--rsi-oversold", type=int, help="RSI oversold niveau")
    parser.add_argument("--rsi-overbought", type=int, help="RSI overbought niveau")
    parser.add_argument("--volatility-factor", type=float,
        help="Volatiliteit factor voor filter")

    # Risk parameters
    parser.add_argument("--capital", type=float, default=10000,
        help="Initieel kapitaal")
    parser.add_argument("--risk", type=float, default=0.01,
        help="Risico per trade (bijv. 0.01 voor 1%)")
    parser.add_argument("--max-daily-loss", type=float, default=0.05,
        help="Maximaal dagelijks verlies (bijv. 0.05 voor 5%)", )
    parser.add_argument("--max-total-loss", type=float, default=0.10,
        help="Maximaal totaal verlies (bijv. 0.10 voor 10%)", )

    # Output opties
    parser.add_argument("--detailed", action="store_true",
        help="Toon gedetailleerde statistieken")
    parser.add_argument("--plot", action="store_true", help="Toon grafieken")
    parser.add_argument("--save-results", action="store_true",
        help="Sla resultaten op naar CSV")
    parser.add_argument("--output-dir", type=str, default="./results",
        help="Output directory voor resultaten", )

    # Optimalisatie
    parser.add_argument("--optimize", action="store_true",
        help="Voer parameteroptimalisatie uit")
    parser.add_argument("--optimize-param", type=str,
        choices=["ema", "rsi", "volatility"], help="Parameter om te optimaliseren", )
    parser.add_argument("--optimize-metric", type=str, default="sharpe",
        choices=["sharpe", "sortino", "calmar", "profit_factor", "win_rate",
            "max_drawdown", ], help="Metric om te optimaliseren", )

    return parser.parse_args()


def calculate_win_rate(trades):
    """
    Berekent de win rate handmatig uit portfolio trades.

    Parameters:
    -----------
    trades : DataFrame
        DataFrame met trade informatie

    Returns:
    --------
    float
        Win rate als percentage
    """
    if len(trades) == 0:
        return 0.0

    winning_trades = len(trades[trades["return"] > 0])
    return winning_trades / len(trades) * 100


def calculate_profit_factor(trades):
    """
    Berekent de profit factor handmatig uit portfolio trades.

    Parameters:
    -----------
    trades : DataFrame
        DataFrame met trade informatie

    Returns:
    --------
    float
        Profit factor (bruto winst / bruto verlies)
    """
    if len(trades) == 0:
        return 0.0

    winning_trades = trades[trades["return"] > 0]
    losing_trades = trades[trades["return"] < 0]

    gross_profit = winning_trades["return"].sum() if len(winning_trades) > 0 else 0
    gross_loss = abs(losing_trades["return"].sum()) if len(losing_trades) > 0 else 0

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def run_backtest(df, strategy_params, risk_params, initial_capital=10000.0,
        detailed=False):
    """
    Voert een vectorized backtest uit op de gegeven data.

    Parameters:
    -----------
    df : pandas.DataFrame
        OHLCV data
    strategy_params : dict
        Strategie parameters
    risk_params : dict
        Risk management parameters
    initial_capital : float
        Initieel kapitaal
    detailed : bool
        Als True, bereken en toon gedetailleerde metrics

    Returns:
    --------
    dict
        Backtest resultaten
    """
    start_time = time.time()
    logger.info(f"Backtest starten met strategie parameters: {strategy_params}")
    logger.info(f"Risk parameters: {risk_params}")

    print(f"\nBacktest gestart om {datetime.now().strftime('%H:%M:%S')}")
    print(f"Genereren van signalen voor {len(df)} bars...", end="", flush=True)

    # Genereer entry/exit signalen
    t0 = time.time()
    entry_signals, exit_signals = multi_layer_ema_strategy(df,
        ema_periods=strategy_params["ema_periods"],
        rsi_period=strategy_params["rsi_period"],
        rsi_oversold=strategy_params["rsi_oversold"],
        rsi_overbought=strategy_params["rsi_overbought"],
        volatility_factor=strategy_params["volatility_factor"], )

    t1 = time.time()
    print(f" Voltooid in {t1 - t0:.2f} seconden")

    print(f"Uitvoeren van portfolio simulatie... ", end="", flush=True)

    # VectorBT portfolio simulatie
    try:
        t0 = time.time()

        # Detecteer frequentie van de index (nieuwe verbeterde methode)
        freq = detect_timeframe_frequency(df.index)

        portfolio = vbt.Portfolio.from_signals(df.close, entries=entry_signals,
            exits=exit_signals, init_cash=initial_capital, fees=0.0001,
            # 1 pip commissie/spread
            freq=freq, )
        t1 = time.time()
        print(f"Voltooid in {t1 - t0:.2f} seconden")
    except Exception as e:
        logger.error(f"Fout bij portfolio simulatie: {str(e)}")
        print(f"\nFout bij portfolio simulatie: {str(e)}")
        return None, None

    # Extracteer trade informatie
    trades = portfolio.trades.records

    # Bereken basisstatistieken
    total_return = portfolio.total_return()

    # Gebruik try/except voor verschillende VectorBT versies
    try:
        sharpe_ratio = portfolio.sharpe_ratio()
    except:
        # Handmatige berekening als fallback
        returns = portfolio.returns()
        sharpe_ratio = (
            returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0)

    try:
        max_drawdown = portfolio.max_drawdown()
    except:
        try:
            # Probeer via value() methode
            value_series = portfolio.value()
            max_dd = (value_series / value_series.cummax() - 1).min()
            max_drawdown = max_dd if not np.isnan(max_dd) else 0
        except:
            # Als dat niet lukt, probeer direct via total_return
            max_drawdown = 0

    # Bereken win rate en profit factor handmatig
    if hasattr(trades, "to_pandas"):
        trades_df = trades.to_pandas()
    else:
        trades_df = pd.DataFrame(trades)

    win_rate = calculate_win_rate(trades_df)
    profit_factor = calculate_profit_factor(trades_df)

    # Verzamel resultaten
    results = {"total_return": float(total_return), "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown), "win_rate": float(win_rate),
        "total_trades": len(trades_df), "profit_factor": float(profit_factor), }

    if hasattr(trades_df, "pnl"):
        pnl_column = "pnl"
    elif hasattr(trades_df, "return"):
        pnl_column = "return"
    else:
        pnl_column = None

    if pnl_column and len(trades_df) > 0:
        winning_trades = trades_df[trades_df[pnl_column] > 0]
        losing_trades = trades_df[trades_df[pnl_column] < 0]

        results["avg_winning_trade"] = (
            winning_trades[pnl_column].mean() if len(winning_trades) > 0 else 0)
        results["avg_losing_trade"] = (
            losing_trades[pnl_column].mean() if len(losing_trades) > 0 else 0)
        results["best_trade"] = trades_df[pnl_column].max() if len(trades_df) > 0 else 0
        results["worst_trade"] = (
            trades_df[pnl_column].min() if len(trades_df) > 0 else 0)

    # Print basisresultaten
    print("\n" + "=" * 50)
    print("          BACKTEST RESULTATEN            ")
    print("=" * 50)
    print(f"Totaal rendement:   {total_return:.2%}")
    print(f"Sharpe ratio:       {sharpe_ratio:.2f}")
    print(f"Maximum drawdown:   {max_drawdown:.2%}")
    print(f"Win rate:           {win_rate:.2f}%")
    print(f"Aantal trades:      {len(trades_df)}")
    print(f"Profit factor:      {profit_factor:.2f}")

    # Bereken en toon gedetailleerde statistieken indien gevraagd
    if detailed:
        print("\n" + "=" * 50)
        print("      GEDETAILLEERDE STATISTIEKEN        ")
        print("=" * 50)

        if pnl_column and len(trades_df) > 0:
            print(f"Gemiddelde winnende trade: {results['avg_winning_trade']:.2f}")
            print(f"Gemiddelde verliezende trade: {results['avg_losing_trade']:.2f}")
            print(f"Beste trade: {results['best_trade']:.2f}")
            print(f"Slechtste trade: {results['worst_trade']:.2f}")

        # FTMO compliance check
        ftmo_risk_manager = FTMORiskManager(initial_capital=initial_capital,
            max_daily_loss_pct=risk_params["max_daily_loss"],
            max_total_loss_pct=risk_params["max_total_loss"], )

        # Simuleer dagelijkse P&L voor FTMO-check
        try:
            daily_returns = portfolio.returns().resample("D").sum()
            max_daily_loss_pct = daily_returns.min()
            daily_loss_breaches = (daily_returns < -risk_params["max_daily_loss"]).sum()

            # Voeg resultaten toe
            results["max_daily_loss_pct"] = float(max_daily_loss_pct)
            results["daily_loss_breaches"] = int(daily_loss_breaches)
            results["ftmo_compliant"] = (max_drawdown <= risk_params[
                "max_total_loss"] and max_daily_loss_pct >= -risk_params[
                "max_daily_loss"] and daily_loss_breaches == 0)

            print("\n" + "=" * 50)
            print("          FTMO COMPLIANCE CHECK          ")
            print("=" * 50)
            print(f"Maximum dagelijks verlies: {max_daily_loss_pct:.2%}")
            print(
                f"Aantal keer dagelijks verlies limiet doorbroken: {daily_loss_breaches}")
            print(f"FTMO compliant: {'JA' if results['ftmo_compliant'] else 'NEE'}")
        except Exception as e:
            logger.warning(f"Kon FTMO compliance niet berekenen: {str(e)}")

    elapsed_time = time.time() - start_time
    print("\n" + "=" * 50)
    print(
        f"Backtest voltooid in {elapsed_time:.2f} seconden ({elapsed_time / 60:.2f} minuten)")
    print("=" * 50)

    # Ruim geheugen op voor efficiëntie
    gc.collect()

    return results, portfolio


def optimize_parameters(df, symbol, base_params, risk_params, param_to_optimize,
        optimization_metric, initial_capital=10000.0, ):
    """
    Voert parameteroptimalisatie uit.

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
    dict
        Optimale parameters en resultaten
    """
    logger.info(
        f"Parameteroptimalisatie starten voor {param_to_optimize}, optimalisatie metric: {optimization_metric}")
    start_time = time.time()

    print(f"\nParameteroptimalisatie gestart voor {param_to_optimize}")
    print(f"Optimalisatie metric: {optimization_metric}")

    optimization_results = []

    # Detecteer frequentie vooraf (één keer)
    freq = detect_timeframe_frequency(df.index)

    if param_to_optimize == "ema":
        # EMA lengtes optimaliseren
        # Definieer een grid van mogelijke EMA combinaties
        ema1_values = [5, 8, 10, 12, 15, 20]
        ema2_values = [20, 30, 40, 50, 60]
        ema3_values = [100, 150, 200, 250]

        # Bereken totaal aantal combinaties
        total_combinations = sum(
            1 for ema1 in ema1_values for ema2 in ema2_values for ema3 in ema3_values if
            ema1 < ema2 and ema2 < ema3)

        print(f"\nTesten van {total_combinations} EMA combinaties...")

        # Maak een tqdm progress bar
        progress_bar = tqdm(total=total_combinations, desc="Optimalisatie voortgang",
            bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}", )

        for ema1 in ema1_values:
            for ema2 in ema2_values:
                for ema3 in ema3_values:
                    if (
                            ema1 < ema2 and ema2 < ema3):  # Zorg dat EMA's in oplopende volgorde zijn
                        test_params = base_params.copy()
                        test_params["ema_periods"] = [ema1, ema2, ema3]

                        # Voer backtest uit met deze parameters
                        try:
                            # Genereer signalen
                            entry_signals, exit_signals = multi_layer_ema_strategy(df,
                                ema_periods=test_params["ema_periods"],
                                rsi_period=test_params["rsi_period"],
                                rsi_oversold=test_params["rsi_oversold"],
                                rsi_overbought=test_params["rsi_overbought"],
                                volatility_factor=test_params["volatility_factor"], )

                            # Simuleer portfolio
                            portfolio = vbt.Portfolio.from_signals(df.close,
                                entries=entry_signals, exits=exit_signals,
                                init_cash=initial_capital, fees=0.0001, freq=freq)

                            # Bereken metrics
                            try:
                                total_return = portfolio.total_return()
                                sharpe_ratio = portfolio.sharpe_ratio()
                                max_drawdown = portfolio.max_drawdown()
                            except:
                                # Fallback voor verschillende versies
                                returns = portfolio.returns()
                                total_return = (
                                                           portfolio.final_value() - initial_capital) / initial_capital
                                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(
                                    252) if returns.std() != 0 else 0
                                max_drawdown = 0

                            # Bereken win rate
                            trades = portfolio.trades.records
                            if hasattr(trades, "to_pandas"):
                                trades_df = trades.to_pandas()
                            else:
                                trades_df = pd.DataFrame(trades)

                            win_rate = calculate_win_rate(trades_df)
                            profit_factor = calculate_profit_factor(trades_df)

                            # Voeg resultaten toe aan lijst
                            optimization_results.append(
                                {"ema1": ema1, "ema2": ema2, "ema3": ema3,
                                    "sharpe_ratio": sharpe_ratio,
                                    "total_return": total_return,
                                    "max_drawdown": max_drawdown, "win_rate": win_rate,
                                    "profit_factor": profit_factor,
                                    "total_trades": len(trades_df), })

                            # Update progress bar met beschrijving van huidige test
                            progress_bar.set_description(
                                f"EMA {ema1}-{ema2}-{ema3}: Sharpe={sharpe_ratio:.2f}")

                            # Ruim onnodige objecten op
                            portfolio = None
                            trades_df = None
                            entry_signals = None
                            exit_signals = None

                            # Voorkom geheugenlekkage na elke 20 optimalisaties
                            if len(optimization_results) % 20 == 0:
                                gc.collect()

                        except Exception as e:
                            logger.error(
                                f"Fout bij optimalisatie van EMA [{ema1}, {ema2}, {ema3}]: {str(e)}")

                        # Update progress bar
                        progress_bar.update(1)

        # Sluit progress bar
        progress_bar.close()

    elif param_to_optimize == "rsi":
        # RSI parameters optimaliseren
        rsi_periods = [5, 8, 10, 14, 20]
        oversold_values = [20, 25, 30, 35, 40]
        overbought_values = [60, 65, 70, 75, 80]

        # Bereken totaal aantal combinaties
        total_combinations = sum(
            1 for rsi_period in rsi_periods for oversold in oversold_values for
            overbought in overbought_values if oversold < overbought)

        print(f"\nTesten van {total_combinations} RSI combinaties...")

        # Maak een tqdm progress bar
        progress_bar = tqdm(total=total_combinations, desc="Optimalisatie voortgang",
            bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}", )

        for rsi_period in rsi_periods:
            for oversold in oversold_values:
                for overbought in overbought_values:
                    if oversold < overbought:  # Logische check
                        test_params = base_params.copy()
                        test_params["rsi_period"] = rsi_period
                        test_params["rsi_oversold"] = oversold
                        test_params["rsi_overbought"] = overbought

                        # Voer backtest uit met deze parameters
                        try:
                            # Genereer signalen
                            entry_signals, exit_signals = multi_layer_ema_strategy(df,
                                ema_periods=test_params["ema_periods"],
                                rsi_period=test_params["rsi_period"],
                                rsi_oversold=test_params["rsi_oversold"],
                                rsi_overbought=test_params["rsi_overbought"],
                                volatility_factor=test_params["volatility_factor"], )

                            # Simuleer portfolio
                            portfolio = vbt.Portfolio.from_signals(df.close,
                                entries=entry_signals, exits=exit_signals,
                                init_cash=initial_capital, fees=0.0001, freq=freq)

                            # Bereken metrics
                            try:
                                total_return = portfolio.total_return()
                                sharpe_ratio = portfolio.sharpe_ratio()
                                max_drawdown = portfolio.max_drawdown()
                            except:
                                # Fallback voor verschillende versies
                                returns = portfolio.returns()
                                total_return = (
                                                           portfolio.final_value() - initial_capital) / initial_capital
                                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(
                                    252) if returns.std() != 0 else 0
                                max_drawdown = 0

                            # Bereken win rate
                            trades = portfolio.trades.records
                            if hasattr(trades, "to_pandas"):
                                trades_df = trades.to_pandas()
                            else:
                                trades_df = pd.DataFrame(trades)

                            win_rate = calculate_win_rate(trades_df)
                            profit_factor = calculate_profit_factor(trades_df)

                            # Voeg resultaten toe aan lijst
                            optimization_results.append(
                                {"rsi_period": rsi_period, "rsi_oversold": oversold,
                                    "rsi_overbought": overbought,
                                    "sharpe_ratio": sharpe_ratio,
                                    "total_return": total_return,
                                    "max_drawdown": max_drawdown, "win_rate": win_rate,
                                    "profit_factor": profit_factor,
                                    "total_trades": len(trades_df), })

                            # Update progress bar met beschrijving van huidige test
                            progress_bar.set_description(
                                f"RSI {rsi_period} ({oversold}/{overbought}): Sharpe={sharpe_ratio:.2f}")

                            # Ruim onnodige objecten op
                            portfolio = None
                            trades_df = None
                            entry_signals = None
                            exit_signals = None

                            # Voorkom geheugenlekkage na elke 20 optimalisaties
                            if len(optimization_results) % 20 == 0:
                                gc.collect()

                        except Exception as e:
                            logger.error(
                                f"Fout bij optimalisatie van RSI [period={rsi_period}, oversold={oversold}, overbought={overbought}]: {str(e)}")

                        # Update progress bar
                        progress_bar.update(1)

        # Sluit progress bar
        progress_bar.close()

    elif param_to_optimize == "volatility":
        # Volatiliteitsfactor optimaliseren
        volatility_factors = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

        print(f"\nTesten van {len(volatility_factors)} volatiliteitsfactoren...")

        # Maak een tqdm progress bar
        progress_bar = tqdm(total=len(volatility_factors),
            desc="Optimalisatie voortgang",
            bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}", )

        for vol_factor in volatility_factors:
            test_params = base_params.copy()
            test_params["volatility_factor"] = vol_factor

            # Voer backtest uit met deze parameters
            try:
                # Genereer signalen
                entry_signals, exit_signals = multi_layer_ema_strategy(df,
                    ema_periods=test_params["ema_periods"],
                    rsi_period=test_params["rsi_period"],
                    rsi_oversold=test_params["rsi_oversold"],
                    rsi_overbought=test_params["rsi_overbought"],
                    volatility_factor=test_params["volatility_factor"], )

                # Simuleer portfolio
                portfolio = vbt.Portfolio.from_signals(df.close, entries=entry_signals,
                    exits=exit_signals, init_cash=initial_capital, fees=0.0001,
                    freq=freq)

                # Bereken metrics
                try:
                    total_return = portfolio.total_return()
                    sharpe_ratio = portfolio.sharpe_ratio()
                    max_drawdown = portfolio.max_drawdown()
                except:
                    # Fallback voor verschillende versies
                    returns = portfolio.returns()
                    total_return = (
                                               portfolio.final_value() - initial_capital) / initial_capital
                    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(
                        252) if returns.std() != 0 else 0
                    max_drawdown = 0

                # Bereken win rate
                trades = portfolio.trades.records
                if hasattr(trades, "to_pandas"):
                    trades_df = trades.to_pandas()
                else:
                    trades_df = pd.DataFrame(trades)

                win_rate = calculate_win_rate(trades_df)
                profit_factor = calculate_profit_factor(trades_df)

                # Voeg resultaten toe aan lijst
                optimization_results.append(
                    {"volatility_factor": vol_factor, "sharpe_ratio": sharpe_ratio,
                        "total_return": total_return, "max_drawdown": max_drawdown,
                        "win_rate": win_rate, "profit_factor": profit_factor,
                        "total_trades": len(trades_df), })

                # Update progress bar met beschrijving van huidige test
                progress_bar.set_description(
                    f"Volatiliteit {vol_factor}: Sharpe={sharpe_ratio:.2f}")

                # Ruim onnodige objecten op
                portfolio = None
                trades_df = None
                entry_signals = None
                exit_signals = None

            except Exception as e:
                logger.error(
                    f"Fout bij optimalisatie van volatiliteitsfactor [{vol_factor}]: {str(e)}")

            # Update progress bar
            progress_bar.update(1)

        # Sluit progress bar
        progress_bar.close()

    # Converteer naar DataFrame voor makkelijkere analyse
    results_df = pd.DataFrame(optimization_results)

    # Voer garbage collection uit
    gc.collect()

    # Sorteer op optimalisatie metric
    if len(results_df) == 0:
        logger.error("Geen geldige optimalisatieresultaten gevonden.")
        print("\nGeen geldige optimalisatieresultaten gevonden.")
        return None, None, None

    if optimization_metric == "sharpe":
        results_df = results_df.sort_values("sharpe_ratio", ascending=False)
    elif optimization_metric == "sortino":
        # Sortino is niet direct beschikbaar, gebruik Sharpe als proxy
        results_df = results_df.sort_values("sharpe_ratio", ascending=False)
    elif optimization_metric == "calmar":
        # Calmar = annualized return / max drawdown
        results_df["calmar"] = (
                results_df["total_return"] / results_df["max_drawdown"].abs())
        results_df = results_df.sort_values("calmar", ascending=False)
    elif optimization_metric == "profit_factor":
        results_df = results_df.sort_values("profit_factor", ascending=False)
    elif optimization_metric == "win_rate":
        results_df = results_df.sort_values("win_rate", ascending=False)
    elif optimization_metric == "max_drawdown":
        results_df = results_df.sort_values("max_drawdown", ascending=True)

    # Neem de beste parameters
    best_params = results_df.iloc[0].to_dict()

    # Print de top 5 resultaten
    print("\n" + "=" * 60)
    print("        OPTIMALISATIERESULTATEN - TOP 5              ")
    print("=" * 60)
    if len(results_df) >= 5:
        print(results_df.head(5).to_string())
    else:
        print(results_df.to_string())

    # Bijwerken van de optimale parameters
    optimal_params = base_params.copy()
    if param_to_optimize == "ema":
        optimal_params["ema_periods"] = [int(best_params["ema1"]),
            int(best_params["ema2"]), int(best_params["ema3"]), ]
    elif param_to_optimize == "rsi":
        optimal_params["rsi_period"] = int(best_params["rsi_period"])
        optimal_params["rsi_oversold"] = int(best_params["rsi_oversold"])
        optimal_params["rsi_overbought"] = int(best_params["rsi_overbought"])
    elif param_to_optimize == "volatility":
        optimal_params["volatility_factor"] = float(best_params["volatility_factor"])

    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(
        f"Optimalisatie voltooid in {elapsed_time:.2f} seconden ({elapsed_time / 60:.2f} minuten)")
    print("=" * 60)

    return optimal_params, best_params, results_df


def plot_backtest_results(df, portfolio, symbol, timeframe):
    """Plot backtest resultaten."""
    print(f"\nPlotten van backtest resultaten voor {symbol} ({timeframe})...")

    try:
        # Maak subplot figuur
        fig, axs = plt.subplots(3, 1, figsize=(12, 16),
            gridspec_kw={"height_ratios": [2, 1, 1]})

        # Plot 1: Prijs en trades
        ax1 = axs[0]
        ax1.plot(df.index, df["close"], label="Close Price", color="blue", alpha=0.6)

        # Plot entry/exit signalen indien beschikbaar
        if hasattr(portfolio, "entries") and hasattr(portfolio, "exits"):
            entries = portfolio.entries.values
            exits = portfolio.exits.values

            for i in range(len(entries)):
                if i < len(df.index) and entries[i]:
                    ax1.scatter(df.index[i], df["close"].iloc[i], color="green",
                        marker="^", s=100, )
                if i < len(df.index) and exits[i]:
                    ax1.scatter(df.index[i], df["close"].iloc[i], color="red",
                        marker="v", s=100)

        # Plot indicators
        if "ema_short" in df.columns:
            ax1.plot(df.index, df["ema_short"], label=f"EMA kort", color="red",
                alpha=0.7)
        if "ema_medium" in df.columns:
            ax1.plot(df.index, df["ema_medium"], label=f"EMA middel", color="blue",
                alpha=0.7)
        if "ema_long" in df.columns:
            ax1.plot(df.index, df["ema_long"], label=f"EMA lang", color="black",
                alpha=0.7)
        if "volatility_band" in df.columns:
            ax1.plot(df.index, df["volatility_band"], label="Volatility Band",
                color="purple", linestyle="--", alpha=0.7, )

        ax1.set_title(f"{symbol} {timeframe} - Prijs en Handelssignalen")
        ax1.set_ylabel("Prijs")
        ax1.legend()
        ax1.grid(True)

        # Plot 2: RSI
        ax2 = axs[1]
        if "rsi" in df.columns:
            ax2.plot(df.index, df["rsi"], label="RSI", color="blue")
            ax2.axhline(y=70, color="r", linestyle="-", alpha=0.3)
            ax2.axhline(y=30, color="g", linestyle="-", alpha=0.3)
            ax2.axhline(y=50, color="black", linestyle="--", alpha=0.2)
            ax2.fill_between(df.index, 70, 100, color="red", alpha=0.1)
            ax2.fill_between(df.index, 0, 30, color="green", alpha=0.1)
            ax2.set_title("RSI Indicator")
            ax2.set_ylabel("RSI")
            ax2.set_ylim(0, 100)
            ax2.grid(True)

        # Plot 3: Equity curve
        ax3 = axs[2]
        try:
            portfolio.plot(ax=ax3)
        except:
            # Fallback voor andere VectorBT versies
            try:
                equity = portfolio.value()
                ax3.plot(equity.index, equity.values, label="Equity")
                ax3.set_title("Equity Curve")
                ax3.set_ylabel("Portfolio Value")
                ax3.legend()
                ax3.grid(True)
            except:
                logger.warning("Kon equity curve niet plotten")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        logger.error(f"Fout bij plotten van resultaten: {str(e)}")
        print(f"Fout bij plotten van resultaten: {str(e)}")


def save_results(results, portfolio, symbol, timeframe, output_dir="./results"):
    """Slaat backtest resultaten op."""
    print(f"\nResultaten opslaan naar {output_dir}...")

    try:
        # Maak output directory indien deze niet bestaat
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Timestamp voor unieke bestandsnamen
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Sla samenvattende resultaten op als JSON
        results_file = os.path.join(output_dir,
            f"{symbol}_{timeframe}_{timestamp}_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4, default=str)

        # Probeer trades op te slaan als CSV
        try:
            # Verschillende VectorBT versies hebben andere trade record attributen
            if hasattr(portfolio.trades.records, "to_pandas"):
                trades_df = portfolio.trades.records.to_pandas()
            else:
                trades_df = pd.DataFrame(portfolio.trades.records)

            trades_file = os.path.join(output_dir,
                f"{symbol}_{timeframe}_{timestamp}_trades.csv")
            trades_df.to_csv(trades_file, index=False)
        except Exception as e:
            logger.warning(f"Kon trades niet opslaan: {str(e)}")

        # Sla equity curve op als CSV
        try:
            equity_file = os.path.join(output_dir,
                f"{symbol}_{timeframe}_{timestamp}_equity.csv")
            portfolio.value().to_csv(equity_file)
        except Exception as e:
            logger.warning(f"Kon equity curve niet opslaan: {str(e)}")

        logger.info(f"Resultaten opgeslagen in {output_dir}")
        print(f"Resultaten opgeslagen in {output_dir}")

    except Exception as e:
        logger.error(f"Fout bij opslaan resultaten: {str(e)}")
        print(f"Fout bij opslaan resultaten: {str(e)}")


def main():
    """Hoofdfunctie voor backtesting."""
    overall_start_time = time.time()
    args = parse_args()

    # Toon banner
    print("\n" + "=" * 80)
    print("""
    Sophy3 - Backtesting Tool v3.0
    """)
    print("=" * 80)

    # Initialiseer MT5 indien credentials zijn opgegeven
    if args.mt5_account and args.mt5_password:
        print(f"Verbinden met MetaTrader 5...")
        if not initialize_mt5(args.mt5_account, args.mt5_password, args.mt5_server):
            logger.error("MT5 initialisatie gefaald, afbreken.")
            print("MT5 initialisatie gefaald, afbreken.")
            return
        print("Verbonden met MetaTrader 5")

    # Bepaal start/eind datums
    end_date = (
        datetime.now() if args.end_date is None else datetime.strptime(args.end_date,
                                                                       "%Y-%m-%d"))

    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    else:
        # Schatting op basis van bars en timeframe
        days = args.bars
        if args.timeframe.startswith("M"):
            # Minuut timeframe
            mins = int(args.timeframe[1:]) if len(args.timeframe) > 1 else 1
            days = int(args.bars * mins / (60 * 24)) + 1  # +1 voor marge
        elif args.timeframe.startswith("H"):
            # Uur timeframe
            hours = int(args.timeframe[1:]) if len(args.timeframe) > 1 else 1
            days = int(args.bars * hours / 24) + 1  # +1 voor marge
        elif args.timeframe.startswith("D"):
            # Dag timeframe
            days = args.bars

        start_date = end_date - timedelta(days=days * 2)  # *2 voor weekends/holidays

    logger.info(
        f"Backtesting {args.symbol} op {args.timeframe} van {start_date} tot {end_date}")
    print(f"Backtesting {args.symbol} op {args.timeframe}")
    print(
        f"Periode: {start_date.strftime('%Y-%m-%d')} tot {end_date.strftime('%Y-%m-%d')}")

    # Haal data op
    print(f"Data ophalen voor {args.symbol}...", end="", flush=True)
    t0 = time.time()
    df = get_data(args.symbol, args.timeframe, start_date, end_date,
        use_cache=not args.no_cache, refresh_cache=args.refresh_cache, )
    t1 = time.time()
    print(f" Voltooid in {t1 - t0:.2f} seconden")

    if df is None or len(df) == 0:
        logger.error("Geen data ontvangen, afbreken.")
        print("Geen data ontvangen, afbreken.")
        return

    logger.info(f"Data ontvangen: {len(df)} bars van {df.index[0]} tot {df.index[-1]}")
    print(
        f"Data ontvangen: {len(df)} bars van {df.index[0].strftime('%Y-%m-%d %H:%M')} tot {df.index[-1].strftime('%Y-%m-%d %H:%M')}")

    # Haal parameters op
    if args.parameter_preset:
        logger.info(f"Gebruik parameter preset: {args.parameter_preset}")
        print(f"Gebruik parameter preset: {args.parameter_preset}")
        asset_class = args.parameter_preset
    else:
        asset_class = (
            args.symbol)  # get_strategy_params detecteert automatisch de asset class
        print(f"Auto-detectie asset class voor {args.symbol}...")

    strategy_params = get_strategy_params(asset_class)
    risk_params = get_risk_params(asset_class)

    # Overwrite parameters indien opgegeven via command line
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

    # Parameter optimalisatie indien gevraagd
    if args.optimize and args.optimize_param:
        strategy_params, best_params, results_df = optimize_parameters(df, args.symbol,
            strategy_params, risk_params, args.optimize_param, args.optimize_metric,
            args.capital, )

        # Controleer of optimalisatie succesvol was
        if strategy_params is None:
            logger.error("Parameteroptimalisatie mislukt, afbreken.")
            print("Parameteroptimalisatie mislukt, afbreken.")
            return

        # Sla optimalisatieresultaten op indien gevraagd
        if args.save_results and results_df is not None:
            output_dir = os.path.join(args.output_dir, "optimization")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_df.to_csv(os.path.join(output_dir,
                f"{args.symbol}_{args.timeframe}_{args.optimize_param}_{timestamp}.csv", ))

    # Voer backtest uit met finale parameters
    results, portfolio = run_backtest(df, strategy_params, risk_params, args.capital,
        args.detailed)

    # Controleer of backtest succesvol was
    if results is None or portfolio is None:
        logger.error("Backtest mislukt, afbreken.")
        print("Backtest mislukt, afbreken.")
        return

    # Plot resultaten indien gevraagd
    if args.plot:
        plot_backtest_results(df, portfolio, args.symbol, args.timeframe)

    # Sla resultaten op indien gevraagd
    if args.save_results:
        save_results(results, portfolio, args.symbol, args.timeframe, args.output_dir)

    # Sluit MT5 verbinding
    shutdown_mt5()

    # Toon totale uitvoeringstijd
    overall_elapsed_time = time.time() - overall_start_time
    print("\n" + "=" * 80)
    print(
        f"Totale uitvoeringstijd: {overall_elapsed_time:.2f} seconden ({overall_elapsed_time / 60:.2f} minuten)")
    print("=" * 80)


if __name__ == "__main__":
    main()