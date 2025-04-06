# src/utils/time_utils.py
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
import logging

logger = logging.getLogger(__name__)

def detect_timeframe_frequency(df_index):
    """
    Detecteert pandas frequentie string op basis van index.

    Parameters:
    -----------
    df_index : pandas.DatetimeIndex
        DatetimeIndex van het DataFrame

    Returns:
    --------
    str or None
        Pandas-compatibele frequentie string of None bij ongeldige invoer

    Voorbeelden:
    --------
    >>> index = pd.date_range(start='2025-01-01', periods=10, freq='H')
    >>> detect_timeframe_frequency(index)
    '1H'
    >>> index = pd.date_range(start='2025-01-01', periods=1)
    >>> detect_timeframe_frequency(index)
    None
    """
    if not isinstance(df_index, pd.DatetimeIndex) or len(df_index) < 2:
        logger.error("Ongeldige index voor frequentiedetectie: geen DatetimeIndex of te weinig datapunten")
        return None

    # Gebruik pd.infer_freq voor robuuste detectie
    freq = pd.infer_freq(df_index)
    if freq:
        return freq
    else:
        logger.warning("Kon frequentie niet detecteren met pd.infer_freq, fallback naar '1H'")
        return "1H"  # Fallback