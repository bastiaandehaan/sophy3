import os
import pandas as pd
import pyarrow.parquet as pq
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

CACHE_DIR = "data/cache"

def get_cache_path(symbol: str, timeframe: str, asset_class: str = None) -> str:
    filename = f"{symbol}_{timeframe}.parquet"
    if asset_class:
        filename = f"{asset_class}_{filename}"
    return os.path.join(CACHE_DIR, filename)

def is_cache_valid(cache_path: str, max_age: timedelta) -> bool:
    if not os.path.exists(cache_path):
        return False
    file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
    return datetime.now() - file_mod_time <= max_age

def get_cache_max_age(timeframe: str) -> timedelta:
    return timedelta(days=30)  # Aanpassen indien nodig

def load_from_cache(symbol: str, timeframe: str, asset_class: str = None,
                    start_date=None, end_date=None):
    cache_path = get_cache_path(symbol, timeframe, asset_class)
    max_age = get_cache_max_age(timeframe)

    if not is_cache_valid(cache_path, max_age):
        logger.info(f"Geen geldige cache gevonden voor {symbol} {timeframe}")
        return None

    try:
        # Laad Parquet-bestand
        table = pq.read_table(cache_path)
        df = table.to_pandas()

        # Controleer of de kolom 'time' aanwezig is en converteer deze
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
