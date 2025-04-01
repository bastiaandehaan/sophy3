"""
Sophy3 - Multi-layer EMA Strategy
Functie: Implementatie van een vectorized Multi-layer EMA trading strategie met volatiliteitsfilter
Auteur: AI Trading Assistant
Laatste update: 2025-04-01

Gebruik:
  Deze module bevat de kern van de trading strategie en kan worden gebruikt
  voor zowel backtesting als live trading. 

Dependencies:
  - vectorbt
  - numpy
  - pandas
"""

import numpy as np
import pandas as pd

def ema(series, length):
    """Berekent de Exponential Moving Average van een tijdreeks."""
    return series.ewm(span=length, adjust=False).mean()

def rsi(series, length):
    """Berekent de Relative Strength Index van een tijdreeks."""
    # Bereken dagelijkse veranderingen
    delta = series.diff()

    # Maak positieve en negatieve verandering series
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Bereken gemiddelde gain en loss
    avg_gain = gain.rolling(window=length).mean()
    avg_loss = loss.rolling(window=length).mean()

    # Bereken relatieve sterkte (RS)
    rs = avg_gain / avg_loss

    # Bereken RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi

def stdev(series, length):
    """Berekent de standaarddeviatie van een tijdreeks."""
    return series.rolling(window=length).std()

def multi_layer_ema_strategy(df, ema_periods=[20, 50, 200], rsi_period=14,
                            rsi_oversold=30, rsi_overbought=70,
                            volatility_factor=0.5):
    """
    Een enkele strategie-implementatie voor backtesting én live trading.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame met OHLCV data (moet 'open', 'high', 'low', 'close', 'volume' kolommen bevatten)
    ema_periods : list
        Lijst met periodes voor de drie EMA's [korte, middel, lange]
    rsi_period : int
        Periode voor RSI berekening
    rsi_oversold : int
        RSI niveau voor oversold conditie (entry)
    rsi_overbought : int
        RSI niveau voor overbought conditie (exit)
    volatility_factor : float
        Factor voor volatiliteitsfilter (standaardwaarde: 0.5)

    Returns:
    --------
    tuple
        (entry_signals, exit_signals) als pandas.Series met boolean waarden
    """
    # Valideer input data
    required_columns = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame moet de volgende kolommen bevatten: {required_columns}")

    # Maak een kopie om te voorkomen dat we het origineel wijzigen
    df = df.copy()

    # Snelle, middel en lange termijn EMA's (volledig vectorized berekening)
    df['ema_short'] = ema(df['close'], ema_periods[0])
    df['ema_medium'] = ema(df['close'], ema_periods[1])
    df['ema_long'] = ema(df['close'], ema_periods[2])

    # RSI voor momentum bevestiging
    df['rsi'] = rsi(df['close'], rsi_period)

    # Volatiliteitsfilter (standaarddeviatie over zelfde periode als langste EMA)
    df['std_dev'] = stdev(df['close'], ema_periods[2])
    df['volatility_band'] = df['ema_long'] + volatility_factor * df['std_dev']

    # Genereer signalen - volledig vectorized (geen loops)
    # Entry: Alle EMAs in oplopende volgorde (trend), prijs boven volatiliteitsband, RSI niet overbought
    entry_conditions = [
        df['ema_short'] > df['ema_medium'],  # Korte termijn > middellange termijn
        df['ema_medium'] > df['ema_long'],   # Middellange termijn > lange termijn
        df['close'] > df['volatility_band'],  # Prijs boven volatiliteitsband
        df['rsi'] < rsi_overbought     # Niet overbought
    ]

    # Optioneel: extra bevestiging voor sterkere signalen
    # bullish_momentum = df['rsi'] > 50  # Bullish momentum
    # entry_conditions.append(bullish_momentum)

    # Exit: Trend breekt (korte EMA kruist onder middellange EMA) of RSI overbought
    exit_conditions = [
        df['ema_short'] < df['ema_medium'],  # Trend breekt
        df['rsi'] > rsi_overbought     # Overbought
    ]

    # Combineer condities met logische operatoren
    entry_signals = pd.Series(True, index=df.index)
    for condition in entry_conditions:
        entry_signals &= condition

    exit_signals = pd.Series(False, index=df.index)
    for condition in exit_conditions:
        exit_signals |= condition

    # Vul NaN waarden in (eerste bars hebben geen EMA, STD waarden)
    entry_signals = entry_signals.fillna(False)
    exit_signals = exit_signals.fillna(False)

    return entry_signals, exit_signals

def calculate_position_size(capital: float, risk_per_trade: float,
                           entry_price: float, stop_loss_price: float) -> float:
    """
    Berekent de positiegrootte op basis van risico per trade.

    Parameters:
    -----------
    capital : float
        Beschikbaar handelskapitaal
    risk_per_trade : float
        Percentage risico per trade (bijv. 0.01 voor 1%)
    entry_price : float
        Verwachte entry prijs
    stop_loss_price : float
        Stop-loss niveau

    Returns:
    --------
    float
        Positiegrootte in basismunteenheid
    """
    if entry_price <= stop_loss_price:
        raise ValueError("Entry price moet hoger zijn dan stop loss voor long posities")

    risk_amount = capital * risk_per_trade
    price_risk = abs(entry_price - stop_loss_price)
    position_size = risk_amount / price_risk

    return position_size