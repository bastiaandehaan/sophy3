# path from content root [data/cache.py]
"""
Sophy3 - Data Cache
Functie: Efficiënte data caching met Parquet
Auteur: AI Trading Assistant
Laatste update: 2025-04-02

Gebruik:
  Deze module verzorgt het cachen en ophalen van historische marktdata om MT5-connectiviteit
  te minimaliseren en backtesting performance te verbeteren.

Dependencies:
  - pandas
  - pyarrow
  - fastparquet
"""

import os
import pandas as pd
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Stel logger in
logger = logging.getLogger(__name__)

# Cache configuratie
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cached_data")
CACHE_VERSION = "v1"  # Increment als cachestructuur verandert
COMPRESSION = "snappy"  # 'snappy' is goed gebalanceerd tussen compressie en snelheid

def setup_cache_dir():
    """Opzetten van cache directory structuur."""
    # Maak basis cache directory
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        logger.info(f"Cache directory aangemaakt: {CACHE_DIR}")

    # Maak asset class subdirectories
    for asset_class in ['forex', 'crypto', 'stocks', 'indices']:
        asset_dir = os.path.join(CACHE_DIR, asset_class)
        if not os.path.exists(asset_dir):
            os.makedirs(asset_dir)
            logger.info(f"Cache subdirectory aangemaakt: {asset_dir}")

def get_cache_path(symbol, timeframe, asset_class=None):
    """
    Bepaalt het pad voor een gecachet databestand.

    Parameters:
    -----------
    symbol : str
        Trading instrument symbool (bijv. EURUSD)
    timeframe : str
        Timeframe als string (bijv. H1)
    asset_class : str, optional
        Asset class ('forex', 'crypto', 'stocks', 'indices')
        Als None, wordt automatisch bepaald

    Returns:
    --------
    str
        Volledig pad naar cache bestand
    """
    # Bepaal asset class indien niet opgegeven
    if asset_class is None:
        # Importeer hier om circulaire imports te voorkomen
        from strategies.params import detect_asset_class
        asset_class = detect_asset_class(symbol)

    # Maak de symbolenmap indien nodig
    symbol_dir = os.path.join(CACHE_DIR, asset_class, symbol)
    if not os.path.exists(symbol_dir):
        os.makedirs(symbol_dir)

    # Bepaal bestandsnaam met versie
    filename = f"{timeframe}_{CACHE_VERSION}.parquet"
    return os.path.join(symbol_dir, filename)

def is_cache_valid(cache_path, max_age_days=1):
    """
    Controleert of een cache bestand geldig en recent genoeg is.

    Parameters:
    -----------
    cache_path : str
        Pad naar cache bestand
    max_age_days : int, optional
        Maximum leeftijd in dagen, afhankelijk van timeframe

    Returns:
    --------
    bool
        True als cache geldig en recent is
    """
    # Check of bestand bestaat
    if not os.path.exists(cache_path):
        return False

    # Check leeftijd
    file_time = os.path.getmtime(cache_path)
    file_date = datetime.fromtimestamp(file_time)
    max_age = timedelta(days=max_age_days)

    # Bestand is te oud als het ouder is dan max_age
    if datetime.now() - file_date > max_age:
        logger.info(f"Cache bestand is verouderd: {cache_path}")
        return False

    return True

def get_cache_max_age(timeframe):
    """
    Bepaalt maximale cache leeftijd op basis van timeframe.

    Parameters:
    -----------
    timeframe : str
        Timeframe als string (bijv. M5, H1, D1)

    Returns:
    --------
    int
        Maximum cache leeftijd in dagen
    """
    timeframe = timeframe.upper()

    # Lagere timeframes vaker refreshen
    if timeframe.startswith('M'):  # Minuut timeframes
        return 1  # Dagelijks vernieuwen
    elif timeframe.startswith('H'):  # Uur timeframes
        return 3  # Elke 3 dagen
    else:  # Dag, week, maand timeframes
        return 7  # Wekelijks

def save_to_cache(df, symbol, timeframe, asset_class=None):
    """
    Slaat DataFrame op in cache.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame met marktdata
    symbol : str
        Trading instrument symbool
    timeframe : str
        Timeframe als string
    asset_class : str, optional
        Asset class categorie

    Returns:
    --------
    bool
        True als succesvol opgeslagen
    """
    if df is None or len(df) == 0:
        logger.warning(f"Geen data om te cachen voor {symbol} {timeframe}")
        return False

    cache_path = get_cache_path(symbol, timeframe, asset_class)

    try:
        # Zorg ervoor dat index een datetime is
        if df.index.name != 'time':
            df = df.reset_index()

        # Sla op als parquet
        df.to_parquet(cache_path, compression=COMPRESSION)
        logger.info(f"Data in cache opgeslagen: {cache_path} ({len(df)} bars)")
        return True
    except Exception as e:
        logger.error(f"Fout bij opslaan in cache: {str(e)}")
        return False

def load_from_cache(symbol, timeframe, asset_class=None, start_date=None, end_date=None):
    """
    Laadt DataFrame uit cache.

    Parameters:
    -----------
    symbol : str
        Trading instrument symbool
    timeframe : str
        Timeframe als string
    asset_class : str, optional
        Asset class categorie
    start_date : datetime, optional
        Begin datum voor data
    end_date : datetime, optional
        Eind datum voor data

    Returns:
    --------
    pandas.DataFrame or None
        DataFrame met marktdata of None als niet gecached/geldig
    """
    cache_path = get_cache_path(symbol, timeframe, asset_class)
    max_age = get_cache_max_age(timeframe)

    if is_cache_valid(cache_path, max_age):
        try:
            # Laad gehele cache
            df = pd.read_parquet(cache_path)

            # Als index geen datetime is, converteer
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'time' in df.columns:
                    df.set_index('time', inplace=True)
                else:
                    logger.warning(f"Geen time kolom in cached data: {cache_path}")
                    return None

            # Filter op datum indien opgegeven
            if start_date is not None:
                df = df[df.index >= start_date]
            if end_date is not None:
                df = df[df.index <= end_date]

            logger.info(f"Data uit cache geladen: {cache_path} ({len(df)} bars)")
            return df
        except Exception as e:
            logger.error(f"Fout bij laden uit cache: {str(e)}")
            return None
    else:
        logger.info(f"Geen geldige cache gevonden voor {symbol} {timeframe}")
        return None

def clear_cache(symbol=None, timeframe=None, asset_class=None, older_than_days=None):
    """
    Verwijdert specifieke cache bestanden.

    Parameters:
    -----------
    symbol : str, optional
        Specifiek symbool om te verwijderen
    timeframe : str, optional
        Specifieke timeframe om te verwijderen
    asset_class : str, optional
        Specifieke asset class om te verwijderen
    older_than_days : int, optional
        Verwijder alleen bestanden ouder dan x dagen

    Returns:
    --------
    int
        Aantal verwijderde bestanden
    """
    cache_base = CACHE_DIR
    deleted_count = 0

    # Bepaal zoekbereik
    if asset_class:
        asset_dirs = [os.path.join(cache_base, asset_class)]
    else:
        asset_dirs = [os.path.join(cache_base, d) for d in
                     os.listdir(cache_base) if os.path.isdir(os.path.join(cache_base, d))]

    cutoff_time = None
    if older_than_days:
        cutoff_time = datetime.now() - timedelta(days=older_than_days)

    for asset_dir in asset_dirs:
        if not os.path.exists(asset_dir):
            continue

        if symbol:
            symbol_dirs = [os.path.join(asset_dir, symbol)]
        else:
            symbol_dirs = [os.path.join(asset_dir, d) for d in
                         os.listdir(asset_dir) if os.path.isdir(os.path.join(asset_dir, d))]

        for symbol_dir in symbol_dirs:
            if not os.path.exists(symbol_dir):
                continue

            for file in os.listdir(symbol_dir):
                file_path = os.path.join(symbol_dir, file)

                # Check timeframe match
                if timeframe and not file.startswith(f"{timeframe}_"):
                    continue

                # Check leeftijd
                if cutoff_time:
                    file_time = os.path.getmtime(file_path)
                    if datetime.fromtimestamp(file_time) > cutoff_time:
                        continue

                # Verwijder bestand
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    logger.info(f"Cache bestand verwijderd: {file_path}")
                except Exception as e:
                    logger.error(f"Fout bij verwijderen cache: {str(e)}")

    return deleted_count

# Initialiseer cache directory bij importeren
setup_cache_dir()