import MetaTrader5 as mt5
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


def initialize_mt5():
    """Initialiseer connectie met MetaTrader 5."""
    if not mt5.initialize():
        logger.error(f"MT5 initialisatie mislukt: {mt5.last_error()}")
        return False
    logger.info("MT5 connectie succesvol.")
    return True


def download_data(symbol: str, timeframe: str, start_date=None, end_date=None) -> \
        Optional[pd.DataFrame]:
    """
    Download data van MetaTrader 5.

    Args:
        symbol: Handelssymbool (bijv. "EURUSD")
        timeframe: Tijdsinterval (bijv. "H1", "D1")
        start_date: Startdatum (optioneel)
        end_date: Einddatum (optioneel)

    Returns:
        DataFrame met OHLCV data of None bij fout
    """
    # MT5 initialiseren als dat nog niet is gebeurd
    if not mt5.terminal_info():
        if not initialize_mt5():
            return None

    # Timeframe mapping
    tf_mapping = {'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5,
                  'M15': mt5.TIMEFRAME_M15, 'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1,
                  'H4': mt5.TIMEFRAME_H4, 'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1, }

    if timeframe not in tf_mapping:
        logger.error(f"Ongeldige timeframe: {timeframe}")
        return None

    # Data periode instellen
    if not end_date:
        end_date = datetime.now()
    if not start_date:
        if timeframe in ['M1', 'M5', 'M15', 'M30']:
            start_date = end_date - timedelta(days=7)
        elif timeframe in ['H1', 'H4']:
            start_date = end_date - timedelta(days=30)
        else:
            start_date = end_date - timedelta(days=365)

    try:
        logger.info(
            f"Ophalen van {symbol} data voor {timeframe} van {start_date} tot {end_date}")

        # Data ophalen van MT5
        rates = mt5.copy_rates_range(symbol, tf_mapping[timeframe], start_date,
                                     end_date)

        if rates is None or len(rates) == 0:
            logger.warning(f"Geen data beschikbaar voor {symbol} {timeframe}")
            return None

        # Converteer naar DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # Sla op als cache
        from data.cache import get_cache_path
        import os
        import pyarrow as pa
        import pyarrow.parquet as pq

        cache_path = get_cache_path(symbol, timeframe)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        # Parquet schrijven
        table = pa.Table.from_pandas(df)
        pq.write_table(table, cache_path)

        logger.info(
            f"Data opgehaald en gecached voor {symbol} {timeframe}: {len(df)} records")
        return df
    except Exception as e:
        logger.error(f"Fout bij data downloaden voor {symbol} {timeframe}: {str(e)}")
        return None