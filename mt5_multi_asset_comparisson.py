"""
MT5 Multi-Asset Data Vergelijker
Vergelijkt historische data voor verschillende assets (forex, crypto, aandelen)
over een periode van 3 jaar op verschillende timeframes.
"""

import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import logging

# Logging configuratie
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Symbolen per categorie (uit FTMO lijst)
SYMBOLS = {'forex': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'],
    'crypto': ['BTCUSD', 'ETHUSD', 'ADAUSD'],
    'stocks': ['AAPL', 'AMZN', 'TSLA', 'MSFT'],
    'indices': ['US500.cash', 'GER40.cash', 'US30.cash']}

# Timeframes voor vergelijking
TIMEFRAMES = {'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15, 'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4, 'D1': mt5.TIMEFRAME_D1}


def initialize_mt5():
    """Initialiseert MT5 verbinding."""
    if not mt5.initialize():
        logger.error(f"MT5 initialisatie mislukt: {mt5.last_error()}")
        return False

    logger.info("MT5 verbinding succesvol geïnitialiseerd")
    return True


def fetch_mt5_data(symbol, timeframe, start_date, end_date):
    """Haalt data op uit MT5 voor de opgegeven periode."""
    logger.info(
        f"Data ophalen voor {symbol} op {timeframe} van {start_date} tot {end_date}")

    # Convert timeframe string to MT5 constant if needed
    mt5_timeframe = TIMEFRAMES.get(timeframe, timeframe)

    # Fetch data
    rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)

    if rates is None or len(rates) == 0:
        logger.warning(f"Geen data ontvangen voor {symbol} op {timeframe}")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    logger.info(f"Data opgehaald voor {symbol} ({timeframe}): {len(df)} bars")
    return df


def analyze_data_quality(df, symbol, timeframe):
    """Analyseert datakwaliteit metrics."""
    if df is None or len(df) == 0:
        logger.warning(f"Geen data om te analyseren voor {symbol} op {timeframe}")
        return None

    # Bereken kwaliteitsmetrics
    metrics = {'symbol': symbol, 'timeframe': timeframe, 'bars_count': len(df),
        'date_range': f"{df.index[0]} tot {df.index[-1]}",
        'duration_days': (df.index[-1] - df.index[0]).days,

        # Prijs metrics
        'avg_spread': (df['high'] - df['low']).mean(),
        'volatility': df['close'].pct_change().std(),

        # Data kwaliteit metrics
        'consecutive_gaps': count_consecutive_gaps(df),
        'price_jumps': detect_price_jumps(df),
        'missing_weekdays': detect_missing_weekdays(df) if timeframe == 'D1' else None,

        # Volume metrics
        'avg_volume': df['tick_volume'].mean(),
        'volume_consistency': df['tick_volume'].std() / df['tick_volume'].mean()}

    return metrics


def count_consecutive_gaps(df):
    """Telt het aantal opeenvolgende tijdsgaten in de data."""
    if len(df) <= 1:
        return 0

    # Calculate expected time difference based on first few intervals
    time_diffs = []
    for i in range(min(5, len(df) - 1)):
        time_diffs.append((df.index[i + 1] - df.index[i]).total_seconds())

    expected_diff = np.median(time_diffs)

    # Count gaps
    gaps = 0
    for i in range(1, len(df)):
        actual_diff = (df.index[i] - df.index[i - 1]).total_seconds()
        if actual_diff > expected_diff * 2:  # More than double expected time
            gaps += 1

    return gaps


def detect_price_jumps(df, threshold=3.0):
    """Detecteert ongewone prijssprongen (> threshold standaarddeviaties)."""
    if len(df) <= 1:
        return 0

    returns = df['close'].pct_change().dropna()
    std_dev = returns.std()
    jumps = (returns.abs() > threshold * std_dev).sum()

    return jumps


def detect_missing_weekdays(df):
    """Detecteert ontbrekende weekdagen in dagelijkse data."""
    if len(df) <= 1:
        return 0

    # Create full range of business days
    start_date = df.index[0]
    end_date = df.index[-1]

    all_days = pd.date_range(start=start_date, end=end_date, freq='B')
    missing_days = len(all_days) - len(df)

    return missing_days


def plot_data_summary(all_metrics):
    """Plot vergelijkende gegevens over dataseries."""
    # Plot data distribution per categorie
    metrics_df = pd.DataFrame(all_metrics)

    # Conversie naar matplotlib-friendly format
    plt.figure(figsize=(15, 10))

    # 1. Verdeling van aantal bars
    plt.subplot(2, 2, 1)
    bars_by_category = metrics_df.groupby(['symbol'])['bars_count'].mean().sort_values(
        ascending=False)
    bars_by_category.plot(kind='bar', title='Gemiddeld aantal bars per symbool')
    plt.tight_layout()

    # 2. Gemiddelde spread per categorie
    plt.subplot(2, 2, 2)
    spread_by_category = metrics_df.groupby(['symbol'])[
        'avg_spread'].mean().sort_values(ascending=False)
    spread_by_category.plot(kind='bar', title='Gemiddelde spread per symbool')
    plt.tight_layout()

    # 3. Gedetecteerde data-issues per timeframe
    plt.subplot(2, 2, 3)
    issues_by_tf = metrics_df.groupby('timeframe')[
        ['consecutive_gaps', 'price_jumps']].mean()
    issues_by_tf.plot(kind='bar', title='Data-issues per timeframe')
    plt.tight_layout()

    # 4. Volumeconsistentie
    plt.subplot(2, 2, 4)
    vol_consistency = metrics_df.groupby('symbol')[
        'volume_consistency'].mean().sort_values(ascending=False)
    vol_consistency.plot(kind='bar',
                         title='Volume consistentie per symbool (lager is beter)')
    plt.tight_layout()

    plt.savefig('mt5_data_quality_summary.png')
    plt.close()

    logger.info("Grafische samenvatting opgeslagen als 'mt5_data_quality_summary.png'")


def main():
    # Initialiseer MT5
    if not initialize_mt5():
        return

    # Bepaal periode (3 jaar)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 3)

    all_metrics = []

    try:
        # Loop door elke symbolencategorie
        for category, symbols in SYMBOLS.items():
            logger.info(f"\n=== Verwerken van {category} symbolen ===")

            for symbol in symbols:
                logger.info(f"Verwerken van {symbol}...")

                for tf_name, tf_value in TIMEFRAMES.items():
                    # Haal data op
                    df = fetch_mt5_data(symbol, tf_value, start_date, end_date)

                    if df is not None and len(df) > 0:
                        # Analyseer datakwaliteit
                        metrics = analyze_data_quality(df, symbol, tf_name)
                        if metrics:
                            metrics['category'] = category
                            all_metrics.append(metrics)

                    # Even pauze om MT5 API niet te overbelasten
                    import time
                    time.sleep(1)

        # Sla metrics op als CSV
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv('mt5_data_quality_metrics.csv', index=False)
        logger.info(
            "Data kwaliteitsmetrics opgeslagen in 'mt5_data_quality_metrics.csv'")

        # Plot samenvattende grafieken
        plot_data_summary(all_metrics)

    finally:
        # Sluit MT5 verbinding
        mt5.shutdown()
        logger.info("MT5 verbinding afgesloten")


if __name__ == "__main__":
    main()