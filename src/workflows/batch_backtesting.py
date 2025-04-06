# src/workflows/batch_backtesting.py
"""
Sophy3 - Batch Backtesting Script (Geoptimaliseerd)
Functie: Test meerdere symbolen en timeframes om prestaties te vergelijken
Auteur: AI Trading Assistant (met input van gebruiker)
Laatste update: 2025-04-06

Gebruik:
  python src/workflows/batch_backtesting.py [--max-memory] [--batch-size 3]

Dependencies:
  - pandas
  - vectorbt
  - psutil (geheugenmonitoring)
  - gc (garbage collection)
  - data.cache
"""

import argparse
import gc
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import psutil
import vectorbt as vbt

# Voeg de root directory toe aan sys.path zodat modules gevonden worden
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
print(f"[INFO] Project root toegevoegd aan sys.path: {root_dir}")

# Sophy3 imports
from src.strategies.ema_strategy import multi_layer_ema_strategy
from src.strategies.params import get_strategy_params, get_risk_params

# Stel logger in
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Default configuratie
DEFAULT_SYMBOLS = ['EURUSD', 'GBPUSD', 'BTCUSD', 'ETHUSD']
DEFAULT_TIMEFRAMES = ['M15', 'H1', 'H4', 'D1']

def log_memory_usage():
    """Log het huidige geheugengebruik."""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)
    logger.info(f"Geheugengebruik: {memory_mb:.2f} MB")
    return memory_mb

def convert_timeframe_to_freq(timeframe):
    """Converteer MT5 timeframe naar pandas frequentie."""
    if timeframe.startswith('M'):
        minutes = timeframe[1:]
        return f"{minutes}T"
    elif timeframe.startswith('H'):
        hours = timeframe[1:]
        return f"{hours}h"
    elif timeframe.startswith('D'):
        days = timeframe[1:]
        return f"{days}D"
    elif timeframe.startswith('W'):
        weeks = timeframe[1:]
        return f"{weeks}W"
    logger.error(f"Onbekend timeframe-formaat: {timeframe}, default naar '1D'")
    return "1D"  # Fallback

def get_data(symbol: str, timeframe: str, max_bars=None):
    """
    Haalt data op uit cache met optimalisatie voor geheugengebruik.

    Parameters:
    -----------
    symbol : str
        Trading instrument symbool
    timeframe : str
        Timeframe als string
    max_bars : int, optional
        Maximum aantal bars om te gebruiken (voor geheugenoptimalisatie)

    Returns:
    --------
    pandas.DataFrame or None
        DataFrame met OHLCV-data
    """
    # Probeer uit cache te laden
    from src.data import get_data
    df = get_data(symbol, timeframe, use_cache=True, refresh_cache=False)
    if df is None:
        logger.error(f"Geen data beschikbaar in cache voor {symbol} {timeframe}")
        return None

    # Beperk het aantal bars indien nodig voor geheugenefficiëntie
    if max_bars is not None and len(df) > max_bars:
        logger.info(f"Beperk {symbol} {timeframe} tot {max_bars} bars voor geheugenefficiëntie")
        return df.tail(max_bars)

    return df

def filter_strategy_params(strategy_params):
    """
    Filtert parameters om alleen de relevante door te geven aan de strategie functie.

    Parameters:
    -----------
    strategy_params : dict
        Alle strategie parameters

    Returns:
    --------
    dict
        Gefilterde parameters die multi_layer_ema_strategy accepteert
    """
    valid_params = ['ema_periods', 'rsi_period', 'rsi_oversold', 'rsi_overbought', 'volatility_factor']
    return {k: v for k, v in strategy_params.items() if k in valid_params}

def calculate_metrics(portfolio, trades_df):
    """
    Bereken performance metrics voor een backtest.

    Parameters:
    -----------
    portfolio : vectorbt.Portfolio
        Portfolio object van backtest
    trades_df : pandas.DataFrame
        DataFrame met trades informatie

    Returns:
    --------
    dict
        Dictionary met performance metrics
    """
    if len(trades_df) == 0:
        logger.warning("Geen trades uitgevoerd in deze backtest")
        return {'sharpe_ratio': 0, 'max_drawdown': 0, 'total_return': 0,
                'win_rate': 0, 'win_loss_ratio': 0, 'total_trades': 0}

    try:
        sharpe_ratio = portfolio.sharpe_ratio()
        max_drawdown = portfolio.max_drawdown()
        total_return = portfolio.total_return()
    except Exception as e:
        logger.warning(f"Fout bij berekenen metrics: {str(e)}")
        returns = portfolio.returns()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
        equity_series = portfolio.equity()
        max_drawdown = ((equity_series.cummax() - equity_series).max() / equity_series.cummax().max()) if equity_series.max() else 0
        total_return = (portfolio.final_value() - portfolio.init_cash) / portfolio.init_cash

    win_rate = len(trades_df[trades_df['return'] > 0]) / len(trades_df) * 100
    avg_win = trades_df[trades_df['return'] > 0]['return'].mean() if len(trades_df[trades_df['return'] > 0]) > 0 else 0
    avg_loss = trades_df[trades_df['return'] < 0]['return'].mean() if len(trades_df[trades_df['return'] < 0]) > 0 else 0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

    return {'sharpe_ratio': sharpe_ratio, 'max_drawdown': max_drawdown,
            'total_return': total_return, 'win_rate': win_rate,
            'win_loss_ratio': win_loss_ratio, 'total_trades': len(trades_df)}

def run_batch_backtesting(symbols=None, timeframes=None, batch_size=None,
                          max_memory_pct=85, adaptive_bars=True):
    """
    Voert batch backtesting uit voor alle symbolen en timeframes met geheugenoptimalisatie.

    Parameters:
    -----------
    symbols : list, optional
        Lijst met te testen symbolen
    timeframes : list, optional
        Lijst met te testen timeframes
    batch_size : int, optional
        Maximum aantal tests om tegelijk uit te voeren (voor geheugenmanagement)
    max_memory_pct : float, optional
        Maximum geheugengebruik percentage (0-100) voordat garbage collection wordt geforceerd
    adaptive_bars : bool, optional
        Pas automatisch het aantal bars aan op basis van timeframe

    Returns:
    --------
    pandas.DataFrame
        DataFrame met resultaten
    """
    if symbols is None:
        symbols = DEFAULT_SYMBOLS
    if timeframes is None:
        timeframes = DEFAULT_TIMEFRAMES

    overall_start_time = time.time()
    logger.info("Batch backtesting gestart")
    print(f"\n[⏱️] Batch backtesting gestart om {datetime.now().strftime('%H:%M:%S')}")
    print(f"[📋] Testing {len(symbols)} symbolen × {len(timeframes)} timeframes = {len(symbols) * len(timeframes)} totale tests")

    os.makedirs('results', exist_ok=True)
    initial_memory = log_memory_usage()

    results = []
    test_count = 0
    batch_count = 0

    if adaptive_bars:
        timeframe_max_bars = {'M1': 5000, 'M5': 5000, 'M15': 3000, 'M30': 2500,
                              'H1': 2000, 'H4': 1000, 'D1': 500, 'W1': 200}
    else:
        timeframe_max_bars = None

    for symbol in symbols:
        for timeframe in timeframes:
            current_memory_pct = log_memory_usage() / (psutil.virtual_memory().total / (1024 * 1024)) * 100
            if current_memory_pct > max_memory_pct:
                logger.warning(f"Geheugengebruik hoog ({current_memory_pct:.1f}%), uitvoeren garbage collection")
                gc.collect()
                log_memory_usage()

            batch_count += 1
            test_count += 1

            if batch_size is not None and batch_count >= batch_size:
                logger.info(f"Batch van {batch_size} tests voltooid, resetten voor geheugenefficiëntie")
                gc.collect()
                batch_count = 0
                log_memory_usage()

            start_time = time.time()
            logger.info(f"Start backtest voor {symbol} op {timeframe}")
            print(f"\n[📈] Backtesting {symbol} op {timeframe}... ({test_count}/{len(symbols) * len(timeframes)})")

            max_bars = timeframe_max_bars.get(timeframe) if adaptive_bars and timeframe in timeframe_max_bars else None
            df = get_data(symbol, timeframe, max_bars)
            if df is None or len(df) == 0:
                logger.warning(f"Geen data beschikbaar voor {symbol} {timeframe}, overslaan")
                print(f"[❌] Geen data beschikbaar, overslaan")
                results.append({'symbol': symbol, 'timeframe': timeframe, 'status': 'Geen data'})
                continue

            logger.info(f"Data voor {symbol} {timeframe}: {len(df)} bars van {df.index[0]} tot {df.index[-1]}")
            print(f"[ℹ️] Data: {len(df)} bars van {df.index[0].strftime('%Y-%m-%d')} tot {df.index[-1].strftime('%Y-%m-%d')}")

            strategy_params = get_strategy_params(symbol)
            risk_params = get_risk_params(symbol)

            try:
                filtered_params = filter_strategy_params(strategy_params)
                entries, exits = multi_layer_ema_strategy(df, **filtered_params)
                pandas_freq = convert_timeframe_to_freq(timeframe)
                portfolio = vbt.Portfolio.from_signals(close=df['close'],
                                                       entries=entries, exits=exits,
                                                       size=1.0,
                                                       freq=pandas_freq)

                trades = portfolio.trades.records
                trades_df = trades.to_pandas() if hasattr(trades, "to_pandas") else pd.DataFrame(trades)
                metrics = calculate_metrics(portfolio, trades_df)

                elapsed_time = time.time() - start_time
                result = {'symbol': symbol, 'timeframe': timeframe,
                          'sharpe_ratio': metrics['sharpe_ratio'],
                          'total_return': metrics['total_return'],
                          'max_drawdown': metrics['max_drawdown'],
                          'win_rate': metrics['win_rate'],
                          'win_loss_ratio': metrics['win_loss_ratio'],
                          'total_trades': metrics['total_trades'],
                          'bars_tested': len(df),
                          'date_range': f"{df.index[0]} tot {df.index[-1]}",
                          'runtime_seconds': elapsed_time}
                results.append(result)

                logger.info(f"Backtest voltooid: {symbol} {timeframe} - Sharpe: {metrics['sharpe_ratio']:.2f}, Return: {metrics['total_return']:.2%}")
                print(f"[✅] Voltooid in {elapsed_time:.2f} seconden - Sharpe: {metrics['sharpe_ratio']:.2f}, Return: {metrics['total_return']:.2%}")

                portfolio = None
                df = None
                entries = None
                exits = None
                gc.collect()

            except Exception as e:
                logger.error(f"Fout bij backtest {symbol} {timeframe}: {str(e)}")
                print(f"[❌] Fout: {str(e)}")
                continue

    if results:
        results_df = pd.DataFrame(results).sort_values('sharpe_ratio', ascending=False)
        print("\n" + "=" * 60)
        print("       Top combinaties van symbolen en timeframes       ")
        print("=" * 60)
        print(results_df.head(10).to_string(index=False))

        print("\n" + "=" * 60)
        print("       Prestatie per symbool (gemiddelde Sharpe ratio)       ")
        print("=" * 60)
        symbol_performance = results_df.groupby('symbol')[['sharpe_ratio', 'total_return', 'max_drawdown']].mean()
        print(symbol_performance.sort_values('sharpe_ratio', ascending=False).to_string())

        print("\n" + "=" * 60)
        print("       Prestatie per timeframe (gemiddelde Sharpe ratio)      ")
        print("=" * 60)
        timeframe_performance = results_df.groupby('timeframe')[['sharpe_ratio', 'total_return', 'max_drawdown']].mean()
        print(timeframe_performance.sort_values('sharpe_ratio', ascending=False).to_string())

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        results_path = f'results/batch_results_{timestamp}.csv'
        results_df.to_csv(results_path, index=False)
        print(f"\n[💾] Resultaten opgeslagen naar: {results_path}")

        overall_elapsed_time = time.time() - overall_start_time
        print(f"[⏱️] Totaal proces voltooid in {overall_elapsed_time:.2f} seconden ({overall_elapsed_time / 60:.2f} minuten)")

        final_memory = log_memory_usage()
        print(f"[🧠] Geheugengebruik: {initial_memory:.1f}MB → {final_memory:.1f}MB")
        return results_df
    else:
        print("\n[❌] Geen resultaten om te tonen.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sophy3 Batch Backtesting Tool")
    parser.add_argument('--symbols', nargs='+', help='Symbolen om te testen (bijv. EURUSD GBPUSD)')
    parser.add_argument('--timeframes', nargs='+', help='Timeframes om te testen (bijv. M15 H1 D1)')
    parser.add_argument('--batch-size', type=int, default=3, help='Aantal tests per batch voor geheugenmanagement')
    parser.add_argument('--max-memory', type=float, default=85, help='Maximum geheugengebruik percentage')
    parser.add_argument('--no-adaptive', action='store_true', help='Schakel adaptieve bar-limiting uit')

    args = parser.parse_args()
    run_batch_backtesting(symbols=args.symbols, timeframes=args.timeframes,
                          batch_size=args.batch_size, max_memory_pct=args.max_memory,
                          adaptive_bars=not args.no_adaptive)