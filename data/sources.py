"""
Sophy3 - Data Sources
Functie: Beheert het ophalen van marktdata uit MT5 of cache
Auteur: AI Trading Assistant
Laatste update: 2025-04-02

Gebruik:
  Beheert het ophalen van marktdata uit MT5 of cache, met ondersteuning voor caching.

Dependencies:
  - MetaTrader5
  - pandas
  - logging
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import logging
from data.cache import load_from_cache, save_to_cache

# Stel logger in
logger = logging.getLogger(__name__)

# MT5 timeframe mapping
MT5_TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}

def initialize_mt5(account=None, password=None, server=None):
    """Initialiseert MT5 terminal."""
    if not mt5.initialize():
        logger.error("MT5 initialisatie gefaald")
        return False
    if account and password:
        if not mt5.login(account, password, server):
            logger.error(f"MT5 login gefaald: {mt5.last_error()}")
            return False
    return True

def shutdown_mt5():
    """Sluit MT5 terminal af."""
    mt5.shutdown()

def get_mt5_data(symbol, timeframe, start_date, end_date):
    """Haalt data op uit MT5."""
    if not mt5.symbol_select(symbol, True):
        logger.error(f"Symbool {symbol} niet gevonden in MT5")
        return None

    # Converteer timeframe naar MT5 formaat
    mt5_timeframe = MT5_TIMEFRAMES.get(timeframe)
    if not mt5_timeframe:
        logger.error(f"Ongeldige timeframe: {timeframe}")
        return None

    # Haal data op
    rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
    if rates is None or len(rates) == 0:
        logger.error(f"Geen data ontvangen van MT5 voor {symbol} {timeframe}")
        return None

    # Converteer naar DataFrame
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df = df[["open", "high", "low", "close", "tick_volume"]]
    df.rename(columns={"tick_volume": "volume"}, inplace=True)

    return df

def get_data(symbol, timeframe, start_date=None, end_date=None, use_cache=True, refresh_cache=False):
    """
    Haalt marktdata op uit cache of MT5.

    Parameters:
    -----------
    symbol : str
        Trading instrument symbool
    timeframe : str
        Timeframe als string (bijv. 'M5', 'H1')
    start_date : datetime, optional
        Start datum
    end_date : datetime, optional
        Eind datum
    use_cache : bool
        Gebruik cache indien beschikbaar
    refresh_cache : bool
        Forceer verversen van cache

    Returns:
    --------
    pandas.DataFrame or None
        DataFrame met marktdata
    """
    # Probeer eerst uit cache te laden
    if use_cache and not refresh_cache:
        df = load_from_cache(symbol, timeframe, start_date=start_date, end_date=end_date)
        if df is not None:
            return df

    # Haal data op via MT5
    if not initialize_mt5():
        logger.error("MT5 initialisatie gefaald, kan geen data ophalen")
        return None

    df = get_mt5_data(symbol, timeframe, start_date, end_date)
    shutdown_mt5()

    if df is not None:
        save_to_cache(df, symbol, timeframe)
        return df

    # Geen fallback naar CSV
    logger.error(f"Geen data beschikbaar voor {symbol} {timeframe}")
    return None