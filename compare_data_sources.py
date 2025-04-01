"""
compare_data_sources.py
Script om MT5- en CSV-data te vergelijken voor een specifiek symbool over meerdere tijdframes.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import MetaTrader5 as mt5

# Stel een logger in
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuratie
CONFIG = {
    "mt5": {
        "enabled": True,
        "path": "C:\\Program Files\\FTMO Global Markets MT5 Terminal\\terminal64.exe",
        "login": None,  # Vul in als nodig
        "password": None,  # Vul in als nodig
        "server": None  # Vul in als nodig
    },
    "data": {
        "csv_directory": "C:\\Users\\Gebruiker\\PycharmProjects\\HIST_DATA",
        "default_timeframe": "H1"
    }
}

# MT5 timeframes mapping
MT5_TIMEFRAMES = {
    'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15, 'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4, 'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1,
    'MN1': mt5.TIMEFRAME_MN1
}

def initialize_mt5():
    """Initialiseert de MT5 connectie."""
    if not mt5.initialize():
        logger.error(f"MT5 initialisatie mislukt: {mt5.last_error()}")
        return False
    if CONFIG["mt5"]["login"] and CONFIG["mt5"]["password"]:
        login_result = mt5.login(
            CONFIG["mt5"]["login"],
            password=CONFIG["mt5"]["password"],
            server=CONFIG["mt5"]["server"]
        )
        if not login_result:
            logger.error(f"MT5 login mislukt: {mt5.last_error()}")
            mt5.shutdown()
            return False
        logger.info(f"MT5 login succesvol voor account {CONFIG['mt5']['login']}")
    return True

def shutdown_mt5():
    """Sluit MT5 connectie af."""
    mt5.shutdown()
    logger.info("MT5 connectie afgesloten")

def fetch_mt5_data(symbol, timeframe, start_date, end_date):
    """Haalt data op uit MT5 voor de opgegeven periode."""
    if not initialize_mt5():
        return None

    mt5_timeframe = MT5_TIMEFRAMES.get(timeframe.upper())
    if mt5_timeframe is None:
        logger.error(f"Ongeldige timeframe: {timeframe}")
        return None

    logger.info(f"MT5 data ophalen voor {symbol} ({timeframe}) van {start_date} tot {end_date}")
    rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)

    if rates is None or len(rates) == 0:
        logger.error(f"Geen data ontvangen voor {symbol} op {timeframe}")
        shutdown_mt5()
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close',
                       'tick_volume': 'volume'}, inplace=True)

    logger.info(f"MT5 data opgehaald voor {symbol} ({timeframe}): {len(df)} bars")
    shutdown_mt5()
    return df

def load_csv_data(symbol, timeframe, csv_dir, start_date, end_date):
    """Laadt data uit een CSV-bestand en filtert op de opgegeven periode."""
    # Probeer eerst het standaardbestand (bijv. EURUSD_H1.csv)
    csv_path = os.path.join(csv_dir, f"{symbol}_{timeframe}.csv")
    if not os.path.exists(csv_path):
        # Als het standaardbestand niet bestaat, probeer een variant (bijv. EURUSD_H1_copy.csv)
        csv_path = os.path.join(csv_dir, f"{symbol}_{timeframe}_copy.csv")
        if not os.path.exists(csv_path):
            logger.warning(f"CSV bestand niet gevonden: {csv_path}")
            return None

    try:
        df = pd.read_csv(csv_path)
        expected_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in expected_columns):
            logger.error(f"CSV bestand mist verwachte kolommen: {csv_path}")
            return None

        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        df = df.loc[start_date:end_date]

        logger.info(f"CSV data opgehaald voor {symbol} ({timeframe}): {len(df)} bars")
        return df
    except Exception as e:
        logger.error(f"Fout bij het laden van CSV bestand: {e}")
        return None

def compare_data(mt5_df, csv_df, symbol, timeframe):
    """Vergelijkt MT5- en CSV-data en rapporteert verschillen."""
    if mt5_df is None or csv_df is None:
        logger.error("Een van de datasets is None, kan niet vergelijken.")
        return

    # Zorg ervoor dat beide DataFrames dezelfde tijdstempels hebben
    common_index = mt5_df.index.intersection(csv_df.index)
    if len(common_index) == 0:
        logger.warning("Geen overlappende tijdstempels gevonden tussen MT5- en CSV-data.")
        return

    mt5_df = mt5_df.loc[common_index]
    csv_df = csv_df.loc[common_index]

    logger.info(f"Vergelijken van {len(common_index)} overlappende bars voor {symbol} ({timeframe})")

    # Vergelijk OHLCV-waarden
    columns = ['open', 'high', 'low', 'close', 'volume']
    differences = {}
    for col in columns:
        diff = mt5_df[col] - csv_df[col]
        abs_diff = diff.abs()
        mean_diff = abs_diff.mean()
        max_diff = abs_diff.max()
        differences[col] = {'mean_diff': mean_diff, 'max_diff': max_diff}

        # Log significante verschillen
        significant_diff = abs_diff[abs_diff > 0.0001]  # Drempel voor prijsverschillen
        if len(significant_diff) > 0:
            logger.warning(f"Significante verschillen in {col}: {len(significant_diff)} bars")
            logger.warning(f"Voorbeeld verschillen:\n{significant_diff.head()}")

    # Samenvatting van verschillen
    logger.info("=== Samenvatting van verschillen ===")
    for col, stats in differences.items():
        logger.info(f"{col}: Gemiddelde afwijking = {stats['mean_diff']:.6f}, Maximale afwijking = {stats['max_diff']:.6f}")

    # Controleer op ontbrekende bars
    mt5_missing = csv_df.index.difference(mt5_df.index)
    csv_missing = mt5_df.index.difference(csv_df.index)
    if len(mt5_missing) > 0:
        logger.warning(f"MT5 mist {len(mt5_missing)} bars die in CSV aanwezig zijn. Eerste ontbrekende: {mt5_missing[0]}")
    if len(csv_missing) > 0:
        logger.warning(f"CSV mist {len(csv_missing)} bars die in MT5 aanwezig zijn. Eerste ontbrekende: {csv_missing[0]}")

def main():
    # Parameters
    symbol = "EURUSD"
    timeframes = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]  # Alle tijdframes om te vergelijken
    start_date = datetime(2020, 1, 1)  # Begin van de periode
    end_date = datetime(2025, 1, 31)   # Einde van de periode (jan 2025)
    csv_dir = CONFIG["data"]["csv_directory"]

    # Vergelijk elk tijdframe
    for timeframe in timeframes:
        logger.info(f"\n=== Vergelijking voor tijdframe {timeframe} ===")
        # Haal data op
        mt5_data = fetch_mt5_data(symbol, timeframe, start_date, end_date)
        csv_data = load_csv_data(symbol, timeframe, csv_dir, start_date, end_date)
        # Vergelijk de data
        compare_data(mt5_data, csv_data, symbol, timeframe)

if __name__ == "__main__":
    main()