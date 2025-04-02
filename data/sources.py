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
from typing import Optional
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
    "W1": mt5.TIMEFRAME_W1,
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

def get_mt5_data(symbol, timeframe, start_date=None, end_date=None, bars=None):
    """
    Haalt data op uit MT5.

    Parameters:
    -----------
    symbol : str
        Handelssymbool (bijv. "EURUSD")
    timeframe : str
        Timeframe als string (bijv. 'M5', 'H1')
    start_date : datetime, optional
        Start datum
    end_date : datetime, optional
        Eind datum
    bars : int, optional
        Aantal bars om op te halen (alternatief voor start_date)

    Returns:
    --------
    pandas.DataFrame or None
        DataFrame met marktdata of None bij fout
    """
    # Controleer of MT5 is geïnitialiseerd
    if not mt5.terminal_info():
        if not initialize_mt5():
            logger.error("MT5 initialisatie gefaald, kan geen data ophalen")
            return None

    # Controleer of symbool bestaat
    if not mt5.symbol_select(symbol, True):
        logger.error(f"Symbool {symbol} niet gevonden in MT5")
        return None

    # Converteer timeframe naar MT5 formaat
    mt5_timeframe = MT5_TIMEFRAMES.get(timeframe)
    if not mt5_timeframe:
        logger.error(f"Ongeldige timeframe: {timeframe}")
        return None

    try:
        logger.info(f"Ophalen van MT5 data voor {symbol} ({timeframe})")

        # Gebruik verschillende methoden afhankelijk van de opgegeven parameters
        if bars is not None:
            # Haal specifiek aantal bars op vanaf het einde
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars)
        elif start_date is not None and end_date is not None:
            # Haal data op tussen start en eind datum
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
        else:
            # Standaard: haal laatste 1000 bars op
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, 1000)

        if rates is None or len(rates) == 0:
            logger.warning(f"Geen data ontvangen van MT5 voor {symbol} {timeframe}")
            return None

        # Converteer naar DataFrame
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)

        # Selecteer en hernoem kolommen voor consistentie
        df = df[["open", "high", "low", "close", "tick_volume"]]
        df.rename(columns={"tick_volume": "volume"}, inplace=True)

        logger.info(f"Data opgehaald voor {symbol} ({timeframe}): {len(df)} bars")
        return df

    except Exception as e:
        logger.error(f"Fout bij data downloaden voor {symbol} {timeframe}: {str(e)}")
        return None

def get_data(symbol, timeframe, start_date=None, end_date=None, use_cache=True,
            refresh_cache=False, bars=None, asset_class=None):
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
    bars : int, optional
        Aantal bars om op te halen (alternatief voor start_date)
    asset_class : str, optional
        Asset class voor betere caching

    Returns:
    --------
    pandas.DataFrame or None
        DataFrame met marktdata
    """
    # Detecteer asset class indien niet opgegeven
    if asset_class is None:
        try:
            from strategies.params import detect_asset_class
            asset_class = detect_asset_class(symbol)
        except ImportError:
            asset_class = None

    # Probeer eerst uit cache te laden
    if use_cache and not refresh_cache:
        df = load_from_cache(symbol, timeframe, asset_class, start_date=start_date, end_date=end_date)
        if df is not None:
            logger.info(f"Data uit cache geladen voor {symbol} ({timeframe}): {len(df)} bars")

            # Beperk het aantal bars indien opgegeven
            if bars is not None and len(df) > bars:
                return df.tail(bars)
            return df

    # Haal data op via MT5
    if not initialize_mt5():
        logger.error("MT5 initialisatie gefaald, kan geen data ophalen")
        return None

    df = get_mt5_data(symbol, timeframe, start_date, end_date, bars)
    shutdown_mt5()

    if df is not None:
        # Sla op in cache
        save_to_cache(df, symbol, timeframe, asset_class)
        return df

    # Geen data beschikbaar
    logger.error(f"Geen data beschikbaar voor {symbol} {timeframe}")
    return None