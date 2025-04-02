# scripts/batch_backtesting.py
"""
Sophy3 - Batch Backtesting Script (Geoptimaliseerd)
Functie: Test meerdere symbolen en timeframes om prestaties te vergelijken
Auteur: AI Trading Assistant (met input van gebruiker)
Laatste update: 2025-04-02

Gebruik:
  python scripts/batch_backtesting.py [--max-memory] [--batch-size 3]

Dependencies:
  - pandas
  - vectorbt
  - psutil (geheugenmonitoring)
  - gc (garbage collection)
  - data.cache
"""

import pandas as pd
import vectorbt as vbt
import numpy as np
import time
import logging
import os
import gc
import psutil
import argparse
from datetime import datetime

# Sophy3 imports
from strategies.multi_layer_ema import multi_layer_ema_strategy
from strategies.params import get_strategy_params, get_risk_params
from data.cache import load_from_cache

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
    df = load_from_cache(symbol, timeframe)
    if df is None:
        logger.error(f"Geen data beschikbaar in cache voor {symbol} {timeframe}")
        return None

    # Beperk het aantal bars indien nodig voor geheugenefficiëntie
    if max_bars is not None and len(df) > max_bars:
        logger.info(
            f"Beperk {symbol} {timeframe} tot {max_bars} bars voor geheugenefficiëntie")
        return df.tail(max_bars)

    return df


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
    # Bereken basismetrics
    try:
        sharpe_ratio = portfolio.sharpe_ratio()
        max_drawdown = portfolio.max_drawdown()
        total_return = portfolio.total_return()
    except Exception as e:
        logger.warning(f"Fout bij berekenen metrics: {str(e)}")
        # Fallback berekeningen
        returns = portfolio.returns()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(
            252) if returns.std() != 0 else 0
        max_drawdown = (
                                   portfolio.equity().cummax() - portfolio.equity()).max() / portfolio.equity().cummax().max()
        total_return = (portfolio.equity().iloc[-1] - portfolio.equity().iloc[0]) / \
                       portfolio.equity().iloc[0]

    # Win rate berekenen
    if len(trades_df) > 0:
        win_rate = len(trades_df[trades_df['return'] > 0]) / len(trades_df) * 100
        avg_win = trades_df[trades_df['return'] > 0]['return'].mean() if len(
            trades_df[trades_df['return'] > 0]) > 0 else 0
        avg_loss = trades_df[trades_df['return'] < 0]['return'].mean() if len(
            trades_df[trades_df['return'] < 0]) > 0 else 0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    else:
        win_rate = 0
        win_loss_ratio = 0

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
    # Standaardwaarden indien niet opgegeven
    if symbols is None:
        symbols = DEFAULT_SYMBOLS
    if timeframes is None:
        timeframes = DEFAULT_TIMEFRAMES

    # Starttijd van het hele proces
    overall_start_time = time.time()
    logger.info("Batch backtesting gestart")
    print(f"\n[⏱️] Batch backtesting gestart om {datetime.now().strftime('%H:%M:%S')}")
    print(
        f"[📋] Testing {len(symbols)} symbolen × {len(timeframes)} timeframes = {len(symbols) * len(timeframes)} totale tests")

    # Zorg dat results directory bestaat
    os.makedirs('results', exist_ok=True)

    # Log initieel geheugengebruik
    initial_memory = log_memory_usage()

    # Resultaten opslaan
    results = []
    test_count = 0
    batch_count = 0

    # Bepaal aantal bars per timeframe voor geheugenefficiëntie
    if adaptive_bars:
        timeframe_max_bars = {'M1': 5000,  # 5000 minuut bars (3-4 dagen)
            'M5': 5000,  # 5000 5-min bars (2-3 weken)
            'M15': 3000,  # 3000 15-min bars (1 maand)
            'M30': 2500,  # 2500 30-min bars (1-2 maanden)
            'H1': 2000,  # 2000 uur bars (3-4 maanden)
            'H4': 1000,  # 1000 4-uur bars (6 maanden)
            'D1': 500,  # 500 dag bars (2 jaar)
            'W1': 200  # 200 week bars (4 jaar)
        }
    else:
        timeframe_max_bars = None

    # Loop door alle symbolen en timeframes
    for symbol in symbols:
        for timeframe in timeframes:
            # Controleer geheugengebruik en forceer garbage collection indien nodig
            current_memory_pct = log_memory_usage() / (
                        psutil.virtual_memory().total / (1024 * 1024)) * 100
            if current_memory_pct > max_memory_pct:
                logger.warning(
                    f"Geheugengebruik hoog ({current_memory_pct:.1f}%), uitvoeren garbage collection")
                gc.collect()
                log_memory_usage()

            # Houd bij hoeveel tests zijn uitgevoerd in de huidige batch
            batch_count += 1
            test_count += 1

            # Reset batch na batch_size tests indien opgegeven
            if batch_size is not None and batch_count >= batch_size:
                logger.info(
                    f"Batch van {batch_size} tests voltooid, resetten voor geheugenefficiëntie")
                gc.collect()
                batch_count = 0
                log_memory_usage()

            # Starttijd van deze specifieke backtest
            start_time = time.time()
            print(
                f"\n[📈] Backtesting {symbol} op {timeframe}... ({test_count}/{len(symbols) * len(timeframes)})")

            # Bepaal maximum aantal bars voor dit timeframe
            max_bars = None
            if adaptive_bars and timeframe in timeframe_max_bars:
                max_bars = timeframe_max_bars[timeframe]

            # Haal data op
            df = get_data(symbol, timeframe, max_bars)
            if df is None or len(df) == 0:
                logger.warning(
                    f"Geen data beschikbaar voor {symbol} {timeframe}, overslaan")
                print(f"[❌] Geen data beschikbaar, overslaan")
                continue

            # Log data info
            logger.info(
                f"Data voor {symbol} {timeframe}: {len(df)} bars van {df.index[0]} tot {df.index[-1]}")
            print(
                f"[ℹ️] Data: {len(df)} bars van {df.index[0].strftime('%Y-%m-%d')} tot {df.index[-1].strftime('%Y-%m-%d')}")

            # Haal strategie- en risicoparameters op
            strategy_params = get_strategy_params(symbol)
            risk_params = get_risk_params(symbol)

            # Genereer signalen
            try:
                entries, exits = multi_layer_ema_strategy(df, **strategy_params)

                # Voer backtest uit met vectorbt
                portfolio = vbt.Portfolio.from_signals(close=df['close'],
                    entries=entries, exits=exits, size=1.0,
                    # Vereenvoudigd: vaste grootte voor vergelijking
                    freq=timeframe  # Timeframe voor juiste berekening
                )

                # Haal trades op
                trades = portfolio.trades.records
                if hasattr(trades, "to_pandas"):
                    trades_df = trades.to_pandas()
                else:
                    trades_df = pd.DataFrame(trades)

                # Bereken metrics
                metrics = calculate_metrics(portfolio, trades_df)

                # Bereken tijd voor deze backtest
                elapsed_time = time.time() - start_time

                # Sla resultaat op
                result = {'symbol': symbol, 'timeframe': timeframe,
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'total_return': metrics['total_return'],
                    'max_drawdown': metrics['max_drawdown'],
                    'win_rate': metrics['win_rate'],
                    'win_loss_ratio': metrics['win_loss_ratio'],
                    'total_trades': metrics['total_trades'], 'bars_tested': len(df),
                    'date_range': f"{df.index[0]} tot {df.index[-1]}",
                    'runtime_seconds': elapsed_time}

                results.append(result)

                logger.info(f"Backtest voltooid: {symbol} {timeframe} - "
                            f"Sharpe: {metrics['sharpe_ratio']:.2f}, Return: {metrics['total_return']:.2%}, "
                            f"Max Drawdown: {metrics['max_drawdown']:.2%}, "
                            f"Tijd: {elapsed_time:.2f}s")
                print(f"[✅] Voltooid in {elapsed_time:.2f} seconden - "
                      f"Sharpe: {metrics['sharpe_ratio']:.2f}, Return: {metrics['total_return']:.2%}, "
                      f"Win Rate: {metrics['win_rate']:.1f}%")

                # Ruim geheugen op
                portfolio = None
                df = None
                entries = None
                exits = None
                gc.collect()

            except Exception as e:
                logger.error(f"Fout bij backtest {symbol} {timeframe}: {str(e)}")
                print(f"[❌] Fout: {str(e)}")
                continue

    # Maak een DataFrame en sorteer op Sharpe ratio
    if results:
        results_df = pd.DataFrame(results).sort_values('sharpe_ratio', ascending=False)

        # Toon de topresultaten
        print("\n" + "=" * 60)
        print("       Top combinaties van symbolen en timeframes       ")
        print("=" * 60)
        print(results_df.head(10).to_string(index=False))

        # Toon samenvatting per asset class
        print("\n" + "=" * 60)
        print("       Prestatie per symbool (gemiddelde Sharpe ratio)       ")
        print("=" * 60)
        symbol_performance = results_df.groupby('symbol')[
            ['sharpe_ratio', 'total_return', 'max_drawdown']].mean()
        print(
            symbol_performance.sort_values('sharpe_ratio', ascending=False).to_string())

        # Toon samenvatting per timeframe
        print("\n" + "=" * 60)
        print("       Prestatie per timeframe (gemiddelde Sharpe ratio)      ")
        print("=" * 60)
        timeframe_performance = results_df.groupby('timeframe')[
            ['sharpe_ratio', 'total_return', 'max_drawdown']].mean()
        print(timeframe_performance.sort_values('sharpe_ratio',
                                                ascending=False).to_string())

        # Sla resultaten op
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        results_path = f'results/batch_results_{timestamp}.csv'
        results_df.to_csv(results_path, index=False)
        print(f"\n[💾] Resultaten opgeslagen naar: {results_path}")

        # Totale tijd van het proces
        overall_elapsed_time = time.time() - overall_start_time
        print("\n" + "=" * 60)
        print(f"[⏱️] Totaal proces voltooid in {overall_elapsed_time:.2f} seconden "
              f"({overall_elapsed_time / 60:.2f} minuten)")

        # Toon geheugengebruik aan het eind
        final_memory = log_memory_usage()
        print(f"[🧠] Geheugengebruik: {initial_memory:.1f}MB → {final_memory:.1f}MB")
        print("=" * 60)

        return results_df
    else:
        print("\n[❌] Geen resultaten om te tonen.")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sophy3 Batch Backtesting Tool")
    parser.add_argument('--symbols', nargs='+',
                        help='Symbolen om te testen (bijv. EURUSD GBPUSD)')
    parser.add_argument('--timeframes', nargs='+',
                        help='Timeframes om te testen (bijv. M15 H1 D1)')
    parser.add_argument('--batch-size', type=int, default=3,
                        help='Aantal tests per batch voor geheugenmanagement')
    parser.add_argument('--max-memory', type=float, default=85,
                        help='Maximum geheugengebruik percentage')
    parser.add_argument('--no-adaptive', action='store_true',
                        help='Schakel adaptieve bar-limiting uit')

    args = parser.parse_args()

    # Voer batch backtesting uit met opgegeven parameters
    run_batch_backtesting(symbols=args.symbols, timeframes=args.timeframes,
        batch_size=args.batch_size, max_memory_pct=args.max_memory,
        adaptive_bars=not args.no_adaptive)