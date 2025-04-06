import logging
import re
import pandas as pd

logger = logging.getLogger(__name__)


def detect_timeframe_frequency(data):
    """
    Try to detect the timeframe frequency from a DataFrame's index

    Args:
        data (DataFrame): DataFrame with a datetime index

    Returns:
        str: Pandas frequency string
    """
    if not isinstance(data, pd.DataFrame):
        logger.warning("Input is not a DataFrame, defaulting to 1H timeframe")
        return 'H'

    if not hasattr(data, 'index') or not hasattr(data.index, 'inferred_freq'):
        logger.warning(
            "DataFrame doesn't have a proper datetime index, defaulting to 1H timeframe")
        return 'H'

    # Try to infer frequency from the index
    freq = pd.infer_freq(data.index)

    if freq is None:
        # If freq couldn't be inferred, try to calculate from consecutive timestamps
        try:
            if len(data.index) >= 2:
                # Calculate time difference between first few rows
                time_diffs = []
                for i in range(min(5, len(data.index) - 1)):
                    diff = data.index[i + 1] - data.index[i]
                    time_diffs.append(diff.total_seconds())

                # Use the most common difference
                most_common_diff = max(set(time_diffs), key=time_diffs.count)

                # Convert seconds to appropriate frequency string
                seconds = most_common_diff
                if seconds == 60:
                    return 'min'
                elif seconds == 300:
                    return '5min'
                elif seconds == 900:
                    return '15min'
                elif seconds == 1800:
                    return '30min'
                elif seconds == 3600:
                    return 'H'
                elif seconds == 14400:
                    return '4H'
                elif seconds == 86400:
                    return 'D'
                elif seconds == 604800:
                    return 'W'
                elif seconds >= 2592000 and seconds <= 2678400:  # ~30 days
                    return 'M'

            logger.warning(
                "Couldn't determine timeframe from index differences, defaulting to 1H")
            return 'H'
        except Exception as e:
            logger.error(f"Error detecting timeframe: {e}")
            return 'H'

    return freq


def parse_timeframe(timeframe_str):
    """
    Parse a timeframe string and return the corresponding pandas frequency

    Args:
        timeframe_str (str): Timeframe string (e.g., '1H', 'D', '15min')

    Returns:
        str: Pandas frequency string
    """
    # If timeframe_str is a DataFrame, extract the actual timeframe from it
    # This likely happens when data is passed directly instead of a timeframe string
    if isinstance(timeframe_str, pd.DataFrame):
        # Try to detect the timeframe from the DataFrame's index
        return detect_timeframe_frequency(timeframe_str)

    # Handle None or empty string
    if not timeframe_str:
        logger.warning("No timeframe specified, defaulting to 1H")
        return 'H'

    # Normalize input: uppercase and remove spaces
    tf = str(timeframe_str).upper().strip()

    # Common timeframe mappings
    timeframe_map = {'M1': 'min', '1M': 'min', 'MIN': 'min', 'MINUTE': 'min',
                     '1MIN': 'min', 'M5': '5min', '5M': '5min', '5MIN': '5min', 'M15': '15min',
                     '15M': '15min', '15MIN': '15min', 'M30': '30min', '30M': '30min',
                     '30MIN': '30min', 'H1': 'H', '1H': 'H', 'HOUR': 'H', 'HOURLY': 'H', 'H4': '4H',
                     '4H': '4H', '4HOUR': '4H', 'D': 'D', 'D1': 'D', '1D': 'D', 'DAY': 'D',
                     'DAILY': 'D', 'W': 'W', 'W1': 'W', '1W': 'W', 'WEEK': 'W', 'WEEKLY': 'W',
                     'MO': 'M', 'M': 'M', 'MN': 'M', 'MN1': 'M', '1MO': 'M', 'MONTH': 'M',
                     'MONTHLY': 'M'}

    # Check if the input matches a known timeframe
    if tf in timeframe_map:
        return timeframe_map[tf]

    # Try to parse more complex formats
    try:
        # Check for patterns like '5M', '15M', '4H'

        # Minutes pattern (e.g., 5M, 15min)
        minutes_match = re.match(r'^(\d+)(M|MIN|MINUTE)S?$', tf)
        if minutes_match:
            count = minutes_match.group(1)
            return f'{count}min'

        # Hours pattern (e.g., 4H, 2hour)
        hours_match = re.match(r'^(\d+)(H|HOUR)S?$', tf)
        if hours_match:
            count = hours_match.group(1)
            if count == '1':
                return 'H'
            return f'{count}H'

        # Days pattern (e.g., 3D, 2day)
        days_match = re.match(r'^(\d+)(D|DAY)S?$', tf)
        if days_match:
            count = days_match.group(1)
            if count == '1':
                return 'D'
            return f'{count}D'

        # Weeks pattern (e.g., 2W, 3week)
        weeks_match = re.match(r'^(\d+)(W|WEEK)S?$', tf)
        if weeks_match:
            count = weeks_match.group(1)
            if count == '1':
                return 'W'
            return f'{count}W'

    except Exception as e:
        logger.error(f"Error parsing timeframe '{timeframe_str}': {e}")

    # Default to hourly if nothing matches
    logger.warning(f"Couldn't parse timeframe '{timeframe_str}', defaulting to 1H")
    return 'H'


def convert_timeframe_to_mt5(timeframe):
    """
    Convert a pandas/python timeframe string to MT5 timeframe constant

    Args:
        timeframe (str): Timeframe string in pandas format (e.g., 'H', '15min', 'D')

    Returns:
        int: MetaTrader 5 timeframe constant
    """
    try:
        import MetaTrader5 as mt5

        # First, normalize the timeframe to a standard format
        tf = parse_timeframe(timeframe)

        # Map pandas frequencies to MT5 timeframe constants
        mt5_map = {'min': mt5.TIMEFRAME_M1, '1min': mt5.TIMEFRAME_M1,
                   '5min': mt5.TIMEFRAME_M5, '15min': mt5.TIMEFRAME_M15,
                   '30min': mt5.TIMEFRAME_M30, 'H': mt5.TIMEFRAME_H1, '1H': mt5.TIMEFRAME_H1,
                   '4H': mt5.TIMEFRAME_H4, 'D': mt5.TIMEFRAME_D1, '1D': mt5.TIMEFRAME_D1,
                   'W': mt5.TIMEFRAME_W1, '1W': mt5.TIMEFRAME_W1, 'M': mt5.TIMEFRAME_MN1,
                   '1M': mt5.TIMEFRAME_MN1}

        if tf in mt5_map:
            return mt5_map[tf]

        # Handle cases where the exact mapping isn't available
        # Extract the numeric part and the unit
        match = re.match(r'^(\d+)([a-zA-Z]+)$', tf)
        if match:
            count = int(match.group(1))
            unit = match.group(2).lower()

            # For minutes
            if unit in ['min', 'minute', 'minutes']:
                if count == 1:
                    return mt5.TIMEFRAME_M1
                elif count == 5:
                    return mt5.TIMEFRAME_M5
                elif count == 15:
                    return mt5.TIMEFRAME_M15
                elif count == 30:
                    return mt5.TIMEFRAME_M30
                else:
                    logger.warning(
                        f"Non-standard minute timeframe: {tf}. Using closest available.")
                    if count < 3:
                        return mt5.TIMEFRAME_M1
                    elif count < 10:
                        return mt5.TIMEFRAME_M5
                    elif count < 20:
                        return mt5.TIMEFRAME_M15
                    else:
                        return mt5.TIMEFRAME_M30

            # For hours
            elif unit in ['h', 'hour', 'hours']:
                if count == 1:
                    return mt5.TIMEFRAME_H1
                elif count == 4:
                    return mt5.TIMEFRAME_H4
                else:
                    logger.warning(
                        f"Non-standard hour timeframe: {tf}. Using closest available.")
                    if count < 2:
                        return mt5.TIMEFRAME_H1
                    else:
                        return mt5.TIMEFRAME_H4

            # For days, weeks, months
            elif unit in ['d', 'day', 'days']:
                return mt5.TIMEFRAME_D1
            elif unit in ['w', 'week', 'weeks']:
                return mt5.TIMEFRAME_W1
            elif unit in ['mo', 'm', 'month', 'months']:
                return mt5.TIMEFRAME_MN1

        # Default to H1 if no match
        logger.warning(
            f"Couldn't convert timeframe '{timeframe}' to MT5 format, defaulting to H1")
        return mt5.TIMEFRAME_H1

    except ImportError:
        logger.error("MetaTrader5 package not installed")
        return None
    except Exception as e:
        logger.error(f"Error converting timeframe to MT5: {e}")
        return None