"""
Sophy3 - Data Sources
Functie: Data toegang & voorbereiding met fallbacks (CSV → MT5 → Mock)
Auteur: AI Trading Assistant
Laatste update: 2025-04-01

Gebruik:
  Deze module verzorgt data toegang vanuit verschillende bronnen met fallback mechanismen.
  Primair via CSV, met fallback naar MT5 en mock data.

Dependencies:
  - pandas
  - numpy
  - MetaTrader5
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Stel een logger in
logger = logging.getLogger(__name__)

# Probeer MT5 te importeren, met fallback als het niet beschikbaar is
try:
    import MetaTrader5 as mt5

    MT5_AVAILABLE = True
except ImportError:
    logger.warning("MetaTrader5 package niet gevonden, fallback naar CSV/mock data.")
    MT5_AVAILABLE = False


def initialize_mt5(account=None, password=None, server=None):
    """
    Initialiseert de MetaTrader5 connectie met optionele login.

    Parameters:
    -----------
    account : int, optional
        MT5 account nummer
    password : str, optional
        MT5 wachtwoord
    server : str, optional
        MT5 server naam

    Returns:
    --------
    bool
        True als initialisatie succesvol is, anders False
    """
    if not MT5_AVAILABLE:
        return False

    # Initialiseer MT5
    if not mt5.initialize():
        logger.error(f"MT5 initialisatie mislukt: {mt5.last_error()}")
        return False

    # Login indien credentials zijn opgegeven
    if account and password:
        login_result = mt5.login(account, password=password, server=server)
        if not login_result:
            logger.error(f"MT5 login mislukt: {mt5.last_error()}")
            mt5.shutdown()
            return False
        logger.info(f"MT5 login succesvol voor account {account} op server {server}")

    return True


def shutdown_mt5():
    """Sluit MT5 connectie veilig af."""
    if MT5_AVAILABLE:
        mt5.shutdown()
        logger.info("MT5 connectie afgesloten")


def get_mt5_timeframe(timeframe_str):
    """
    Converteert timeframe string naar MT5 timeframe constante.

    Parameters:
    -----------
    timeframe_str : str
        Timeframe als string (bijv. "M1", "H1", "D1")

    Returns:
    --------
    int
        MT5 timeframe constante
    """
    if not MT5_AVAILABLE:
        return None

    timeframes = {'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15, 'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4, 'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1,
        'MN1': mt5.TIMEFRAME_MN1}

    return timeframes.get(timeframe_str.upper())


def get_data_from_mt5(symbol, timeframe, start_date=None, num_bars=1000):
    """
    Haalt historische data op van MT5.

    Parameters:
    -----------
    symbol : str
        Instrument symbool (bijv. "EURUSD")
    timeframe : str
        Timeframe als string (bijv. "M15", "H1")
    start_date : datetime, optional
        Begin datum voor data
    num_bars : int, optional
        Aantal bars om op te halen

    Returns:
    --------
    pandas.DataFrame
        DataFrame met OHLCV data of None bij fout
    """
    if not MT5_AVAILABLE:
        logger.warning("MT5 niet beschikbaar voor data, probeer fallback...")
        return None

    # Controleer of MT5 is geïnitialiseerd
    if not mt5.terminal_info():
        if not initialize_mt5():
            return None

    # Converteer timeframe string naar MT5 constante
    mt5_timeframe = get_mt5_timeframe(timeframe)
    if mt5_timeframe is None:
        logger.error(f"Ongeldige timeframe: {timeframe}")
        return None

    # Bepaal startdatum indien niet opgegeven
    if start_date is None:
        start_date = datetime.now() - timedelta(
            days=num_bars / 24 * 2)  # Ruwe schatting

    # Haal data op in vectorized modus (veel sneller dan loop)
    rates = mt5.copy_rates_from(symbol, mt5_timeframe, start_date, num_bars)

    if rates is None or len(rates) == 0:
        logger.error(f"Geen data ontvangen voor {symbol} op {timeframe}")
        return None

    # Converteer naar pandas DataFrame
    df = pd.DataFrame(rates)
    # Converteer 'time' kolom naar datetime (MT5 gebruikt UNIX timestamp)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    # Hernoem kolommen naar standaard formaat
    df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close',
        'tick_volume': 'volume'}, inplace=True)

    logger.info(f"Data opgehaald voor {symbol} ({timeframe}): {len(df)} bars")
    return df


def get_data_from_csv(symbol, timeframe, csv_dir="./data/csv"):
    """
    Haalt data op uit CSV bestand als fallback.

    Parameters:
    -----------
    symbol : str
        Instrument symbool
    timeframe : str
        Timeframe als string
    csv_dir : str, optional
        Directory met CSV bestanden

    Returns:
    --------
    pandas.DataFrame
        DataFrame met OHLCV data of None bij fout
    """
    # Verwachte bestandsnaam format: EURUSD_H1.csv
    csv_path = os.path.join(csv_dir, f"{symbol}_{timeframe}.csv")

    if not os.path.exists(csv_path):
        logger.warning(f"CSV bestand niet gevonden: {csv_path}")
        return None

    try:
        df = pd.read_csv(csv_path)

        # Controleer of het de verwachte kolommen heeft
        expected_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in expected_columns):
            logger.error(f"CSV bestand mist verwachte kolommen: {csv_path}")
            return None

        # Converteer 'time' kolom naar datetime en zet als index
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

        logger.info(
            f"Data opgehaald uit CSV voor {symbol} ({timeframe}): {len(df)} bars")
        return df

    except Exception as e:
        logger.error(f"Fout bij het laden van CSV bestand: {e}")
        return None


def generate_mock_data(symbol, timeframe, num_bars=1000):
    """
    Genereert gesimuleerde OHLCV data als laatste fallback.

    Parameters:
    -----------
    symbol : str
        Instrument symbool
    timeframe : str
        Timeframe als string
    num_bars : int, optional
        Aantal bars om te genereren

    Returns:
    --------
    pandas.DataFrame
        DataFrame met gesimuleerde OHLCV data
    """
    logger.warning(f"Genereren van mock data voor {symbol} ({timeframe})")

    # Bepaal startdatum op basis van timeframe
    now = datetime.now()

    if timeframe == 'M1':
        start_date = now - timedelta(minutes=num_bars)
        freq = 'T'  # minuut
    elif timeframe == 'M5':
        start_date = now - timedelta(minutes=5 * num_bars)
        freq = '5T'
    elif timeframe == 'M15':
        start_date = now - timedelta(minutes=15 * num_bars)
        freq = '15T'
    elif timeframe == 'M30':
        start_date = now - timedelta(minutes=30 * num_bars)
        freq = '30T'
    elif timeframe == 'H1':
        start_date = now - timedelta(hours=num_bars)
        freq = 'H'
    elif timeframe == 'H4':
        start_date = now - timedelta(hours=4 * num_bars)
        freq = '4H'
    elif timeframe == 'D1':
        start_date = now - timedelta(days=num_bars)
        freq = 'D'
    else:
        start_date = now - timedelta(days=num_bars)
        freq = 'D'

    # Genereer tijdreeks
    dates = pd.date_range(start=start_date, periods=num_bars, freq=freq)

    # Genereer prijzen gebaseerd op random walk
    np.random.seed(42)  # Voor reproduceerbaarheid

    # Begin prijs op basis van symbool (voor realisme)
    if 'USD' in symbol:
        if 'EUR' in symbol:
            base_price = 1.1  # EURUSD
        elif 'GBP' in symbol:
            base_price = 1.3  # GBPUSD
        elif 'BTC' in symbol:
            base_price = 35000  # BTCUSD
        else:
            base_price = 100
    else:
        base_price = 100

    # Random walk voor close prijzen
    volatility = 0.01 if 'BTC' not in symbol else 0.03
    close = base_price * np.cumprod(1 + np.random.normal(0, volatility, num_bars))

    # Open, high, low op basis van close
    open_prices = close * (1 + np.random.normal(0, volatility / 2, num_bars))
    high = np.maximum(close, open_prices) * (
                1 + np.abs(np.random.normal(0, volatility, num_bars)))
    low = np.minimum(close, open_prices) * (
                1 - np.abs(np.random.normal(0, volatility, num_bars)))

    # Volume
    volume = np.random.lognormal(10, 1, num_bars)

    # Maak DataFrame
    df = pd.DataFrame({'open': open_prices, 'high': high, 'low': low, 'close': close,
        'volume': volume}, index=dates)

    logger.info(f"Mock data gegenereerd voor {symbol} ({timeframe}): {len(df)} bars")
    return df


def get_data(symbol, timeframe, start_date=None, num_bars=1000, csv_dir="./data/csv"):
    """
    Centrale functie voor data ophalen met fallback mechanismen.

    Parameters:
    -----------
    symbol : str
        Instrument symbool
    timeframe : str
        Timeframe als string
    start_date : datetime, optional
        Begin datum voor data
    num_bars : int, optional
        Aantal bars om op te halen
    csv_dir : str, optional
        Directory met CSV bestanden voor fallback

    Returns:
    --------
    pandas.DataFrame
        DataFrame met OHLCV data
    """
    # Poging 1: CSV (eerst proberen)
    logger.info(f"Proberen CSV-data op te halen voor {symbol} ({timeframe})")
    df = get_data_from_csv(symbol, timeframe, csv_dir)

    # Poging 2: MT5 (als CSV niet werkt)
    if df is None:
        logger.info(f"Fallback naar MT5-data voor {symbol} ({timeframe})")
        df = get_data_from_mt5(symbol, timeframe, start_date, num_bars)

    # Poging 3: Mock data (als laatste resort)
    if df is None:
        logger.warning(f"Fallback naar mock data voor {symbol} ({timeframe})")
        df = generate_mock_data(symbol, timeframe, num_bars)

    return df