"""
Sophy3 - Data Sources
Functie: Data ophalen uit verschillende bronnen
Auteur: AI Trading Assistant
Laatste update: 2025-04-06

Gebruik:
  Deze module bevat functies voor het ophalen van data uit verschillende bronnen.

Dependencies:
  - MetaTrader5
  - pandas
"""

import os
import datetime
import MetaTrader5 as mt5
import pandas as pd
import logging
from enum import Enum
import sys
from pathlib import Path

# Logger setup
logger = logging.getLogger(__name__)

# MetaTrader5 timeframes mapping
MT5_TIMEFRAMES = {'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5,
                  'M15': mt5.TIMEFRAME_M15, 'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1,
                  'H4': mt5.TIMEFRAME_H4, 'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1,
                  'MN1': mt5.TIMEFRAME_MN1}


class DataSourceFactory:
    """
    Factory class to create appropriate data sources
    """

    @staticmethod
    def create_source(source_type, config):
        """
        Create a data source based on source type
        """
        # Accept 'mt5' as a valid source type for historical data
        if source_type in ['historical', 'mt5'] or source_type is None:
            return HistoricalDataSource(config=config)
        # Add other data source types as needed
        else:
            raise ValueError(f"Unknown data source type: {source_type}")

    @staticmethod
    def create_data_source(source_type, config):
        """
        Create a data source based on source type
        """
        # Accept 'mt5' as a valid source type for historical data
        if source_type in ['historical', 'mt5'] or source_type is None:
            return HistoricalDataSource(config=config)
        # Add other data source types as needed
        else:
            raise ValueError(f"Unknown data source type: {source_type}")


class HistoricalDataSource:
    """
    Class to handle historical data retrieval
    """

    def __init__(self, **kwargs):
        # Initialize properties
        self.initialized = False
        self.source = kwargs.get('source', 'mt5')
        self.config = kwargs.get('config', {})

    def get_data(self, symbol, timeframe, start_date, end_date, **kwargs):
        # Use existing get_data function
        return get_data(symbol, timeframe, start_date, end_date, **kwargs)

    def get_historical_data(self, symbol, timeframe, start_date, end_date, **kwargs):
        """
        Alias for get_data to maintain compatibility with the backtest script
        """
        return self.get_data(symbol, timeframe, start_date, end_date, **kwargs)


def initialize_mt5(config=None):
    """
    Initialize the MetaTrader 5 connection

    Args:
        config (dict): Configuration dictionary with MT5 settings

    Returns:
        bool: True if successfully initialized, False otherwise
    """
    if not config:
        config = {}

    try:
        # Initialize MT5
        if not mt5.initialize():
            logger.error(f"MT5 initialization failed. Error code: {mt5.last_error()}")

            # Attempt to identify and handle common initialization issues
            error_code = mt5.last_error()[0]
            if error_code == 10013:  # Permission denied
                logger.error(
                    "Permission denied. Try running as administrator or check firewall settings.")
            elif error_code == 10014:  # Invalid path
                logger.error(
                    "Invalid terminal path. Check the path to your MT5 terminal.")

            return False

        logger.info("MT5 succesvol geïnitialiseerd")

        # If login credentials are provided, attempt to login
        if 'login' in config and 'password' in config and 'server' in config:
            login = config.get('login')
            password = config.get('password')
            server = config.get('server')

            # Optional parameters
            timeout = config.get('timeout', 60000)

            login_result = mt5.login(login, password, server, timeout)

            if not login_result:
                logger.error(f"MT5 login failed. Error code: {mt5.last_error()}")
                mt5.shutdown()
                return False

            logger.info(f"MT5 login successful for account {login}")

        # Display terminal information
        terminal_info = mt5.terminal_info()
        if terminal_info is not None:
            logger.info(f"Terminal name: {terminal_info.name}")
            # Check if version attribute exists before accessing it
            if hasattr(terminal_info, 'version'):
                logger.info(f"Terminal version: {terminal_info.version}")
            logger.info(f"Terminal path: {terminal_info.path}")

        # Display account information if logged in
        account_info = mt5.account_info()
        if account_info is not None:
            logger.info(f"Account: {account_info.login} ({account_info.server})")
            logger.info(f"Account balance: {account_info.balance}")
            logger.info(f"Account equity: {account_info.equity}")

        return True

    except Exception as e:
        logger.error(f"MT5 initialization failed with exception: {e}")
        return False


def shutdown_mt5():
    """
    Shutdown the MetaTrader 5 connection

    Returns:
        bool: True if successfully shut down, False otherwise
    """
    try:
        mt5.shutdown()
        logger.info("MT5 verbinding afgesloten")
        return True
    except Exception as e:
        logger.error(f"MT5 shutdown failed with exception: {e}")
        return False


def get_mt5_data(symbol, timeframe, start_date, end_date, include_volumes=True):
    """
    Get historical data from MetaTrader 5

    Args:
        symbol (str): Symbol to get data for
        timeframe (str): Timeframe to get data for (M1, M5, M15, M30, H1, H4, D1, W1, MN1)
        start_date (datetime): Start date for data retrieval
        end_date (datetime): End date for data retrieval
        include_volumes (bool): Include volume data in the result

    Returns:
        pd.DataFrame: DataFrame with historical data
    """
    # Convert timeframe string to MT5 timeframe
    if timeframe not in MT5_TIMEFRAMES:
        logger.error(f"Invalid timeframe: {timeframe}")
        return None

    mt5_timeframe = MT5_TIMEFRAMES[timeframe]

    # Ensure dates are in datetime format
    if isinstance(start_date, str):
        try:
            start_date = datetime.datetime.fromisoformat(start_date)
        except ValueError:
            try:
                start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                logger.error(f"Invalid start date format: {start_date}")
                return None

    if isinstance(end_date, str):
        try:
            end_date = datetime.datetime.fromisoformat(end_date)
        except ValueError:
            try:
                end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                logger.error(f"Invalid end date format: {end_date}")
                return None

    # Add one day to end_date to include the end date in the query
    end_date = end_date + datetime.timedelta(days=1)

    try:
        # Check if connection is initialized
        if not mt5.terminal_info():
            logger.warning("MT5 not initialized, attempting to initialize...")
            if not initialize_mt5():
                logger.error("Failed to initialize MT5 connection")
                return None

        # Get symbol info to check if the symbol exists
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol {symbol} not found in MT5")
            return None

        # Ensure the symbol is available for data retrieval
        if not symbol_info.visible:
            logger.warning(
                f"Symbol {symbol} is not visible, trying to make it visible...")
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select symbol {symbol}")
                return None

        # Request data
        logger.info(
            f"Requesting data for {symbol} on {timeframe} from {start_date} to {end_date}")
        rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)

        if rates is None or len(rates) == 0:
            logger.warning(
                f"No data returned for {symbol} on {timeframe} from {start_date} to {end_date}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(rates)

        # Convert time to datetime
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # Rename columns
        df = df.rename(
            columns={'time': 'datetime', 'open': 'open', 'high': 'high', 'low': 'low',
                     'close': 'close', 'tick_volume': 'volume', 'spread': 'spread',
                     'real_volume': 'real_volume'})

        # Set datetime as index
        df.set_index('datetime', inplace=True)

        # Remove volume data if not requested
        if not include_volumes:
            df = df.drop(['volume', 'real_volume', 'spread'], axis=1, errors='ignore')

        logger.info(f"Data opgehaald voor {symbol} ({timeframe}): {len(df)} bars")
        return df

    except Exception as e:
        logger.error(f"Error retrieving data from MT5: {e}")
        return None


def get_data(symbol, timeframe, start_date, end_date, source='mt5', **kwargs):
    """
    Get historical data from specified source

    Args:
        symbol (str): Symbol to get data for
        timeframe (str): Timeframe to get data for
        start_date (datetime or str): Start date for data retrieval
        end_date (datetime or str): End date for data retrieval
        source (str): Data source ('mt5', 'csv', etc.)
        **kwargs: Additional source-specific parameters

    Returns:
        pd.DataFrame: DataFrame with historical data
    """
    if source == 'mt5':
        return get_mt5_data(symbol, timeframe, start_date, end_date,
                            include_volumes=kwargs.get('include_volumes', True))
    elif source == 'csv':
        # Implementation for CSV data source
        file_path = kwargs.get('file_path')
        if not file_path:
            # Try to find CSV in default location
            data_folder = kwargs.get('data_folder', 'data')
            filename = f"{symbol}_{timeframe}.csv"
            file_path = os.path.join(data_folder, filename)

        if not os.path.exists(file_path):
            logger.error(f"CSV file not found: {file_path}")
            return None

        try:
            # Read CSV file
            df = pd.read_csv(file_path)

            # Convert datetime column
            date_col = kwargs.get('date_column', 'datetime')
            df[date_col] = pd.to_datetime(df[date_col])

            # Filter by date range
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)

            df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]

            # Set datetime as index
            df.set_index(date_col, inplace=True)

            return df

        except Exception as e:
            logger.error(f"Error reading CSV data: {e}")
            return None
    else:
        logger.error(f"Unsupported data source: {source}")
        return None