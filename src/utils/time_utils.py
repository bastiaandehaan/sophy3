"""
Sophy3 - Time Utilities
Functie: Hulpfuncties voor tijd- en frequentiegerelateerde operaties
Auteur: [Jouw naam]
Laatste update: 2025-04-07

Gebruik:
  Deze module bevat utility functies voor de interpretatie van tijd- en frequentiegegevens.

Dependencies:
  - pandas
"""

import pandas as pd

def detect_timeframe_frequency(df_index):
    """
    Detecteert pandas frequentie string op basis van index.

    Parameters:
    -----------
    df_index : pandas.DatetimeIndex
        DatetimeIndex van het DataFrame

    Returns:
    --------
    str
        Pandas-compatibele frequentie string
    """
    if not isinstance(df_index, pd.DatetimeIndex) or len(df_index) < 2:
        return "1H"

    diffs = [(df_index[i + 1] - df_index[i]) for i in range(min(100, len(df_index) - 1))]
    most_common_diff = pd.Series(diffs).mode()[0].total_seconds()

    if most_common_diff == 60:
        return "1min"
    elif most_common_diff == 300:
        return "5min"
    elif most_common_diff == 900:
        return "15min"
    elif most_common_diff == 1800:
        return "30min"
    elif most_common_diff == 3600:
        return "1H"
    elif most_common_diff == 14400:
        return "4H"
    elif most_common_diff == 86400:
        return "1D"
    elif most_common_diff == 604800:
        return "1W"
    return f"{int(most_common_diff)}S"