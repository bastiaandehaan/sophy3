"""
Sophy3 - Time Utilities
Functie: Hulpfuncties voor tijd- en frequentiegerelateerde operaties
"""

import pandas as pd

def detect_timeframe_frequency(df_index):
    """
    Detecteert pandas frequentie string op basis van index.
    ...
    """
    # Kopieer de functie uit backtest.py