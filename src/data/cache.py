# src/data/cache.py
import logging
import os
from datetime import datetime, timedelta

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

CACHE_DIR = "data/cache"

def get_cache_path(symbol: str, timeframe: str, asset_class: str = None) -> str:
    """
    Bepaalt het pad naar het cachebestand.

    Parameters:
    -----------
    symbol : str
        Trading instrument symbool
    timeframe : str
        Timeframe als string
    asset_class : str, optional
        Asset class (forex, crypto, etc.)

    Returns:
    --------
    str
        Pad naar cachebestand
    """
    # Maak asset-class specifieke subdirectories
    if asset_class:
        directory = os.path.join(CACHE_DIR, asset_class.lower(), symbol)
    else:
        directory = os.path.join(CACHE_DIR, symbol)

    # Zorg dat directory bestaat
    os.makedirs(directory, exist_ok=True)

    filename = f"{timeframe}.parquet"
    return os.path.join(directory, filename)

def is_cache_valid(cache_path: str, max_age: timedelta) -> bool:
    """
    Controleert of cache nog geldig is.

    Parameters:
    -----------
    cache_path : str
        Pad naar cachebestand
    max_age : timedelta
        Maximale geldigheidsduur

    Returns:
    --------
    bool
        True als cache geldig is, anders False
    """
    if not os.path.exists(cache_path):
        return False
    file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
    return datetime.now() - file_mod_time <= max_age

def get_cache_max_age(timeframe: str) -> timedelta:
    """
    Bepaalt maximale geldigheidsduur voor een timeframe.

    Parameters:
    -----------
    timeframe : str
        Timeframe als string

    Returns:
    --------
    timedelta
        Maximale geldigheidsduur
    """
    # Verschillende bewaartijden per timeframe
    if timeframe in ['M1', 'M5', 'M15', 'M30']:
        return timedelta(days=7)  # Kortere periodes: 1 week geldig
    elif timeframe in ['H1', 'H4']:
        return timedelta(days=30)  # Uur-timeframes: 1 maand geldig
    else:
        return timedelta(days=90)  # Dag en hoger: 3 maanden geldig

def save_to_cache(df: pd.DataFrame, symbol: str, timeframe: str, asset_class: str = None) -> bool:
    """
    Slaat DataFrame op in cache.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame met OHLCV data
    symbol : str
        Trading instrument symbool
    timeframe : str
        Timeframe als string
    asset_class : str, optional
        Asset class (forex, crypto, etc.)

    Returns:
    --------
    bool
        True als succesvol opgeslagen, anders False
    """
    cache_path = get_cache_path(symbol, timeframe, asset_class)

    try:
        # Controleer of de df een index heeft die kan worden omgezet naar een kolom
        df_to_save = df.copy()

        # Als de index een DatetimeIndex is, reset deze en voorkom kolomconflicten
        if isinstance(df_to_save.index, pd.DatetimeIndex):
            if 'time' in df_to_save.columns:
                df_to_save = df_to_save.rename(columns={'time': 'original_time'})
            df_to_save = df_to_save.reset_index().rename(columns={'index': 'time'})

        # Maak een PyArrow-tabel en sla op als Parquet met compressie
        table = pa.Table.from_pandas(df_to_save)
        pq.write_table(table, cache_path, compression='snappy')

        logger.info(f"Data in cache opgeslagen: {cache_path} ({len(df)} bars)")
        return True
    except Exception as e:
        logger.error(f"Fout bij opslaan in cache: {str(e)}")
        return False

def load_from_cache(symbol: str, timeframe: str, asset_class: str = None, start_date=None, end_date=None):
    """
    Laadt data uit cache.

    Parameters:
    -----------
    symbol : str
        Trading instrument symbool
    timeframe : str
        Timeframe als string
    asset_class : str, optional
        Asset class (forex, crypto, etc.)
    start_date : datetime, optional
        Start datum voor filtering
    end_date : datetime, optional
        Eind datum voor filtering

    Returns:
    --------
    pandas.DataFrame or None
        DataFrame met OHLCV data of None bij fout
    """
    cache_path = get_cache_path(symbol, timeframe, asset_class)
    max_age = get_cache_max_age(timeframe)

    if not is_cache_valid(cache_path, max_age):
        logger.info(f"Geen geldige cache gevonden voor {symbol} {timeframe}")
        return None

    try:
        # Laad Parquet-bestand
        table = pq.read_table(cache_path)
        df = table.to_pandas()

        # Controleer en stel index in
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
        else:
            logger.warning(f"Geen 'time' kolom in cachebestand: {cache_path}")
            return None

        # Filter op start- en einddatum
        if start_date:
            start_date = pd.Timestamp(start_date)
            df = df[df.index >= start_date]
        if end_date:
            end_date = pd.Timestamp(end_date)
            df = df[df.index <= end_date]

        logger.info(f"Data uit cache geladen: {cache_path} ({len(df)} bars)")
        return df
    except Exception as e:
        logger.error(f"Fout bij laden uit cache: {str(e)}")
        return None

def clear_cache(symbol: str = None, timeframe: str = None, asset_class: str = None, older_than_days: int = None) -> int:
    """
    Verwijdert cachebestanden op basis van filters.

    Parameters:
    -----------
    symbol : str, optional
        Specifiek symbool om te verwijderen
    timeframe : str, optional
        Specifieke timeframe om te verwijderen
    asset_class : str, optional
        Specifieke asset class om te verwijderen
    older_than_days : int, optional
        Verwijder alleen bestanden ouder dan X dagen

    Returns:
    --------
    int
        Aantal verwijderde bestanden
    """
    count = 0

    # Bepaal de basis directory om te doorzoeken
    if asset_class:
        search_dir = os.path.join(CACHE_DIR, asset_class.lower())
    else:
        search_dir = CACHE_DIR

    # Controleer of directory bestaat
    if not os.path.exists(search_dir):
        logger.warning(f"Cache directory bestaat niet: {search_dir}")
        return 0

    # Loop door alle bestanden in de cache directory
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file.endswith('.parquet'):
                file_path = os.path.join(root, file)

                # Check symbol filter (controleer of het pad het symbool bevat)
                if symbol and symbol not in root:
                    continue

                # Check timeframe filter
                if timeframe and not file.startswith(f"{timeframe}."):
                    continue

                # Check leeftijd filter
                if older_than_days:
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if (datetime.now() - file_time).days < older_than_days:
                        continue

                # Verwijder bestand als het voldoet aan alle filters
                try:
                    os.remove(file_path)
                    count += 1
                    logger.info(f"Cache verwijderd: {file_path}")
                except Exception as e:
                    logger.error(f"Fout bij verwijderen cache: {file_path}, {str(e)}")

    return count