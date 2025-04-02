"""
Sophy3 - Data Cache
Functie: Efficiënte data caching met Parquet
Auteur: AI Trading Assistant
Laatste update: 2025-04-02

Gebruik:
  Beheert caching van marktdata in Parquet-formaat voor snelle toegang.

Dependencies:
  - pandas
  - pyarrow
  - logging
"""

import os
import pandas as pd
import pyarrow.parquet as pq
import logging
from datetime import datetime, timedelta
from pathlib import Path
from strategies.params import detect_asset_class

# Stel logger in
logger = logging.getLogger(__name__)

# Cache configuratie
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "cached_data")
CACHE_VERSION = "v1"
COMPRESSION = "snappy"


def setup_cache_dir():
    """Creëert de cache directory structuur als die nog niet bestaat."""
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)


def get_cache_path(symbol: str, timeframe: str, asset_class: str = None) -> str:
    """
    Genereert het pad naar het cachebestand.

    Parameters:
    -----------
    symbol : str
        Trading instrument symbool
    timeframe : str
        Timeframe als string
    asset_class : str, optional
        Asset class categorie, automatisch gedetecteerd als None

    Returns:
    --------
    str
        Volledig pad naar cachebestand
    """
    if asset_class is None:
        asset_class = detect_asset_class(symbol)
    return os.path.join(CACHE_DIR, asset_class, symbol,
                        f"{timeframe}_{CACHE_VERSION}.parquet")


def get_cache_max_age(timeframe: str) -> int:
    """
    Bepaalt maximale leeftijd van cache in dagen gebaseerd op timeframe.

    Parameters:
    -----------
    timeframe : str
        Timeframe als string (bijv. 'M5', 'H1', 'D1')

    Returns:
    --------
    int
        Maximaal aantal dagen
    """
    timeframe_map = {'M1': 1, 'M5': 1, 'M15': 1, 'M30': 2, 'H1': 3, 'H4': 5, 'D1': 7}
    return timeframe_map.get(timeframe, 7)  # Default naar 7 dagen


def is_cache_valid(cache_path: str, max_age_days: int) -> bool:
    """
    Controleert of een cachebestand bestaat en niet te oud is.

    Parameters:
    -----------
    cache_path : str
        Pad naar cachebestand
    max_age_days : int
        Maximale leeftijd in dagen

    Returns:
    --------
    bool
        True als cache geldig is, anders False
    """
    if not os.path.exists(cache_path):
        return False
    file_mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
    age = (datetime.now() - file_mtime).days
    return age <= max_age_days


def save_to_cache(df: pd.DataFrame, symbol: str, timeframe: str,
                  asset_class: str = None):
    """
    Slaat DataFrame op in cache als Parquet-bestand.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame om op te slaan
    symbol : str
        Trading instrument symbool
    timeframe : str
        Timeframe als string
    asset_class : str, optional
        Asset class categorie
    """
    cache_path = get_cache_path(symbol, timeframe, asset_class)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    try:
        df.to_parquet(cache_path, compression=COMPRESSION)
        logger.info(f"Data opgeslagen in cache: {cache_path}")
    except Exception as e:
        logger.error(f"Fout bij opslaan in cache: {str(e)}")
        return False
    return True


def load_from_cache(symbol: str, timeframe: str, asset_class: str = None,
                    start_date=None, end_date=None):
    """
    Laadt DataFrame uit cache met optionele datumfilters.

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

    if not is_cache_valid(cache_path, max_age):
        logger.info(f"Geen geldige cache gevonden voor {symbol} {timeframe}")
        return None

    try:
        # Gebruik pyarrow om gefilterde data te laden
        filters = []
        if start_date:
            filters.append(('time', '>=', pd.Timestamp(start_date)))
        if end_date:
            filters.append(('time', '<=', pd.Timestamp(end_date)))

        table = pq.read_table(cache_path, filters=filters)
        df = table.to_pandas()

        if not isinstance(df.index, pd.DatetimeIndex):
            if 'time' in df.columns:
                df.set_index('time', inplace=True)
            else:
                logger.warning(f"Geen time kolom in cached data: {cache_path}")
                return None

        logger.info(f"Data uit cache geladen: {cache_path} ({len(df)} bars)")
        return df
    except Exception as e:
        logger.error(f"Fout bij laden uit cache: {str(e)}")
        return None


def clear_cache(symbol: str = None, timeframe: str = None):
    """
    Verwijdert cachebestanden.

    Parameters:
    -----------
    symbol : str, optional
        Specifiek symbool om te wissen
    timeframe : str, optional
        Specifieke timeframe om te wissen
    """
    if symbol and timeframe:
        cache_path = get_cache_path(symbol, timeframe)
        if os.path.exists(cache_path):
            os.remove(cache_path)
            logger.info(f"Cache verwijderd: {cache_path}")
    else:
        import shutil
        if symbol:
            asset_class = detect_asset_class(symbol)
            path = os.path.join(CACHE_DIR, asset_class, symbol)
        else:
            path = CACHE_DIR
        if os.path.exists(path):
            shutil.rmtree(path)
            logger.info(f"Cache directory verwijderd: {path}")


# Start setup
setup_cache_dir()