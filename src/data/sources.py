# src/data/sources.py
import logging

import MetaTrader5 as mt5
import pandas as pd

from src.data.cache import load_from_cache, save_to_cache

logger = logging.getLogger(__name__)

MT5_TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1,
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
    logger.info("MT5 succesvol geïnitialiseerd")
    return True

def shutdown_mt5():
    """Sluit MT5 terminal af."""
    mt5.shutdown()
    logger.info("MT5 verbinding afgesloten")

def get_mt5_data(symbol, timeframe, start_date=None, end_date=None, bars=None):
    """
    Haalt data op uit MT5.

    Parameters:
    -----------
    symbol : str
        Handelssymbool
    timeframe : str
        Timeframe (bijv. 'M5', 'H1')
    start_date : datetime, optional
        Start datum
    end_date : datetime, optional
        Eind datum
    bars : int, optional
        Aantal bars (alternatief voor start_date)

    Returns:
    --------
    pandas.DataFrame or None
    """
    mt5_timeframe = MT5_TIMEFRAMES.get(timeframe)
    if not mt5_timeframe:
        logger.error(f"Ongeldige timeframe: {timeframe}")
        return None

    # Controleer symboolbeschikbaarheid
    if not mt5.symbol_info(symbol):
        logger.error(f"Symbool {symbol} niet beschikbaar in MT5")
        return None
    if not mt5.symbol_select(symbol, True):
        logger.error(f"Symbool {symbol} niet geselecteerd in MT5")
        return None

    try:
        if bars is not None:
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars)
        elif start_date and end_date:
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
        else:
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, 1000)
        if rates is None or len(rates) == 0:
            logger.warning(f"Geen data ontvangen van MT5 voor {symbol} {timeframe}")
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df = df[["open", "high", "low", "close", "tick_volume"]]
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        logger.info(f"Data opgehaald voor {symbol} ({timeframe}): {len(df)} bars")
        return df
    except Exception as e:
        logger.error(f"Fout bij data downloaden voor {symbol} {timeframe}: {str(e)}")
        return None

def get_data(symbol, timeframe, start_date=None, end_date=None, use_cache=True,
             refresh_cache=False, bars=None, asset_class=None, manage_connection=True):
    """
    Haalt marktdata op uit cache of MT5.

    Parameters:
    -----------
    symbol : str
        Handelssymbool
    timeframe : str
        Timeframe (bijv. 'M5', 'H1')
    start_date : datetime, optional
        Start datum
    end_date : datetime, optional
        Eind datum
    use_cache : bool
        Gebruik cache indien beschikbaar
    refresh_cache : bool
        Forceer verversen van cache
    bars : int, optional
        Aantal bars (alternatief voor start_date)
    asset_class : str, optional
        Asset class voor caching
    manage_connection : bool
        Indien True, beheert MT5-verbinding intern; indien False, verwacht dat MT5 al actief is

    Returns:
    --------
    pandas.DataFrame or None
    """
    if manage_connection:
        if not initialize_mt5():
            logger.error("MT5 initialisatie gefaald")
            return None
    else:
        if not mt5.terminal_info():
            logger.error("MT5 is niet geïnitialiseerd")
            return None

    if use_cache and not refresh_cache:
        df = load_from_cache(symbol, timeframe, asset_class, start_date=start_date, end_date=end_date)
        if df is not None:
            logger.info(f"Data uit cache geladen voor {symbol} ({timeframe}): {len(df)} bars")
            if bars is not None and len(df) > bars:
                return df.tail(bars)
            return df

    df = get_mt5_data(symbol, timeframe, start_date, end_date, bars)
    if df is not None:
        save_to_cache(df, symbol, timeframe, asset_class)
        if manage_connection:
            shutdown_mt5()
        return df

    if manage_connection:
        shutdown_mt5()
    return None