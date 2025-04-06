# src/strategies/params.py
"""
Sophy3 - Strategy Parameters
Functie: Asset-class specifieke parameter detectie en configuratie
Auteur: AI Trading Assistant
Laatste update: 2025-04-01

Gebruik:
  Deze module verzorgt automatische asset class detectie en bijbehorende parameter selectie
  voor verschillende markten en instrumenten.

Dependencies:
  - geen externe dependencies
"""

def detect_asset_class(symbol: str) -> str:
    """
    Detecteert de asset class gebaseerd op symbol naam.

    Parameters:
    -----------
    symbol : str
        Trading instrument symbool (bijv. "EURUSD", "BTCUSD", "AAPL", etc.)

    Returns:
    --------
    str
        Gedetecteerde asset class ('forex', 'crypto', 'index', of 'stock')
    """
    forex_currencies = ['USD', 'EUR', 'JPY', 'GBP', 'AUD', 'CAD', 'CHF', 'NZD']
    if any(forex in symbol for forex in forex_currencies):
        if len(symbol) <= 6:
            return 'forex'

    crypto_symbols = ['BTC', 'ETH', 'XRP', 'LTC', 'BCH', 'ADA', 'DOT', 'LINK']
    if any(crypto in symbol for crypto in crypto_symbols):
        return 'crypto'

    index_symbols = ['SPX', 'NDX', 'DJI', 'DAX', 'FTSE', 'NKY']
    if any(index in symbol for index in index_symbols):
        return 'index'

    return 'stock'

def get_strategy_params(asset_class: str) -> dict:
    """
    Geeft strategieparameters terug per asset class.

    Parameters:
    -----------
    asset_class : str
        Asset-klasse ('forex', 'crypto', 'index', of 'stock')

    Returns:
    --------
    dict
        Dictionary met strategieparameters aangepast voor de specifieke asset class
    """
    if asset_class == 'forex':
        return {
            'ema_periods': [20, 50, 200],
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'volatility_factor': 0.5
        }
    elif asset_class == 'crypto':
        return {
            'ema_periods': [10, 20, 50],
            'rsi_period': 14,
            'rsi_oversold': 40,
            'rsi_overbought': 60,
            'volatility_factor': 1.5
        }
    elif asset_class == 'index':
        return {
            'ema_periods': [10, 30, 100],
            'rsi_period': 14,
            'rsi_oversold': 40,
            'rsi_overbought': 60,
            'volatility_factor': 0.75
        }
    else:  # stocks
        return {
            'ema_periods': [9, 21, 50],
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'volatility_factor': 0.5
        }

def get_risk_params(asset_class: str) -> dict:
    """
    Haalt risicobeheerparameters op per asset class.

    Parameters:
    -----------
    asset_class : str
        Asset-klasse ('forex', 'crypto', 'index', of 'stock')

    Returns:
    --------
    dict
        Dictionary met risicoparameters voor FTMO compliance
    """
    risk_params = {
        'max_daily_loss': 0.05,
        'max_total_loss': 0.10,
        'profit_target': 0.10,
        'min_trading_days': 4
    }

    if asset_class == 'forex':
        risk_params['risk_per_trade'] = 0.01
    elif asset_class == 'crypto':
        risk_params['risk_per_trade'] = 0.005
        risk_params['max_open_positions'] = 1
    elif asset_class == 'index':
        risk_params['risk_per_trade'] = 0.01
        risk_params['max_open_positions'] = 2
    else:  # stocks
        risk_params['risk_per_trade'] = 0.01
        risk_params['max_open_positions'] = 3

    return risk_params