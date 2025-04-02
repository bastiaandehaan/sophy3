# scripts/batch_backtesting.py
"""
Sophy3 - Batch Backtesting Script
Functie: Test meerdere symbolen en timeframes om prestaties te vergelijken
Auteur: AI Trading Assistant (met input van gebruiker)
Laatste update: 2025-04-02

Gebruik:
  python scripts/batch_backtesting.py

Dependencies:
  - pandas
  - vectorbt
  - data.cache
"""

import pandas as pd
import vectorbt as vbt
from strategies.multi_layer_ema import multi_layer_ema_strategy
from strategies.params import get_strategy_params, get_risk_params
from data.cache import load_from_cache  # Importeer uit data/cache.py
from datetime import datetime
import time
import logging
import os

# Stel logger in
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Lijst van symbolen en timeframes om te testen
symbols = ['EURUSD', 'GBPUSD', 'BTCUSD']  # Voeg je eigen symbolen toe
timeframes = ['H1', 'D1']  # Voeg je eigen timeframes toe


# Functie om data op te halen (alleen cache, geen CSV)
def get_data(symbol: str, timeframe: str):
    """
    Haalt data op uit cache.

    Parameters:
    -----------
    symbol : str
        Trading instrument symbool
    timeframe : str
        Timeframe als string

    Returns:
    --------
    pandas.DataFrame or None
        DataFrame met OHLCV-data
    """
    # Probeer uit cache te laden
    df = load_from_cache(symbol, timeframe)
    if df is not None:
        return df

    # Geen fallback naar CSV; log een fout als cache niet beschikbaar is
    logger.error(f"Geen data beschikbaar in cache voor {symbol} {timeframe}")
    return None


def run_batch_backtesting():
    """Voert batch backtesting uit voor alle symbolen en timeframes."""
    # Starttijd van het hele proces
    overall_start_time = time.time()
    logger.info("Batch backtesting gestart")
    print(f"\n[⏱️] Batch backtesting gestart om {datetime.now().strftime('%H:%M:%S')}")

    # Resultaten opslaan
    results = []

    for symbol in symbols:
        for timeframe in timeframes:
            # Starttijd van deze specifieke backtest
            start_time = time.time()
            print(f"\n[📈] Backtesting {symbol} op {timeframe}...")

            # Haal data op
            df = get_data(symbol, timeframe)
            if df is None or len(df) == 0:
                logger.warning(
                    f"Geen data beschikbaar voor {symbol} {timeframe}, overslaan")
                print(f"[❌] Geen data beschikbaar, overslaan")
                continue

            # Haal strategie- en risicoparameters op
            strategy_params = get_strategy_params(symbol)
            risk_params = get_risk_params(symbol)

            # Genereer signalen
            entries, exits = multi_layer_ema_strategy(df, **strategy_params)

            # Voer backtest uit met vectorbt
            try:
                portfolio = vbt.Portfolio.from_signals(close=df['close'],
                    entries=entries, exits=exits, size=1.0,
                    # Vereenvoudigd: vaste grootte voor vergelijking
                    freq=timeframe  # Timeframe voor juiste berekening
                )

                # Haal prestatie-indicatoren op
                sharpe_ratio = portfolio.sharpe_ratio()
                max_drawdown = portfolio.max_drawdown()

                # Bereken tijd voor deze backtest
                elapsed_time = time.time() - start_time

                # Sla resultaat op
                results.append({'symbol': symbol, 'timeframe': timeframe,
                    'sharpe_ratio': sharpe_ratio, 'max_drawdown': max_drawdown,
                    'runtime_seconds': elapsed_time})

                logger.info(f"Backtest voltooid: {symbol} {timeframe} - "
                            f"Sharpe: {sharpe_ratio:.2f}, Max Drawdown: {max_drawdown:.2%}, "
                            f"Tijd: {elapsed_time:.2f}s")
                print(f"[✅] Voltooid in {elapsed_time:.2f} seconden - "
                      f"Sharpe: {sharpe_ratio:.2f}, Max Drawdown: {max_drawdown:.2%}")

            except Exception as e:
                logger.error(f"Fout bij backtest {symbol} {timeframe}: {str(e)}")
                print(f"[❌] Fout: {str(e)}")
                continue

    # Maak een DataFrame en sorteer op Sharpe ratio
    results_df = pd.DataFrame(results).sort_values('sharpe_ratio', ascending=False)

    # Toon de topresultaten
    print("\n" + "=" * 60)
    print("       Top combinaties van symbolen en timeframes       ")
    print("=" * 60)
    print(results_df.head(10).to_string(index=False))

    # Totale tijd van het proces
    overall_elapsed_time = time.time() - overall_start_time
    print("\n" + "=" * 60)
    print(f"[⏱️] Totaal proces voltooid in {overall_elapsed_time:.2f} seconden "
          f"({overall_elapsed_time / 60:.2f} minuten)")
    print("=" * 60)

    return results_df


if __name__ == "__main__":
    run_batch_backtesting()