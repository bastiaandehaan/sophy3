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
    # Forex detectie
    forex_currencies = ['USD', 'EUR', 'JPY', 'GBP', 'AUD', 'CAD', 'CHF', 'NZD']
    if any(forex in symbol for forex in forex_currencies):
        if len(symbol) <= 6:  # Typische lengte van forex paren
            return 'forex'

    # Crypto detectie
    crypto_symbols = ['BTC', 'ETH', 'XRP', 'LTC', 'BCH', 'ADA', 'DOT', 'LINK']
    if any(crypto in symbol for crypto in crypto_symbols):
        return 'crypto'

    # Index detectie
    index_symbols = ['SPX', 'NDX', 'DJI', 'DAX', 'FTSE', 'NKY']
    if any(index in symbol for index in index_symbols):
        return 'index'

    # Default: stock
    return 'stock'


def get_strategy_params(symbol: str) -> dict:
    """
    Automatische parameterdetectie per asset class.

    Parameters:
    -----------
    symbol : str
        Trading instrument symbool

    Returns:
    --------
    dict
        Dictionary met strategie parameters aangepast voor de specifieke asset class
    """
    asset_class = detect_asset_class(symbol)

    # Basis parameters per asset class
    if asset_class == 'forex':
        return {'ema_periods': [20, 50, 200],  # Maandelijks, kwartaal, jaarlijks
            'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70,
            'volatility_factor': 0.5, 'risk_per_trade': 0.01  # 1% risico per trade
        }
    elif asset_class == 'crypto':
        return {'ema_periods': [10, 20, 50],  # Snellere cycli voor crypto
            'rsi_period': 14, 'rsi_oversold': 40,  # Minder gevoelig voor oversold
            'rsi_overbought': 60,  # Minder gevoelig voor overbought
            'volatility_factor': 1.5,  # Hogere volatiliteit accepteren
            'risk_per_trade': 0.005  # Lager risico (0.5%) vanwege hogere volatiliteit
        }
    elif asset_class == 'index':
        return {'ema_periods': [10, 30, 100],  # Aangepast voor indices
            'rsi_period': 14, 'rsi_oversold': 40, 'rsi_overbought': 60,
            'volatility_factor': 0.75, 'risk_per_trade': 0.01}
    else:  # stocks
        return {'ema_periods': [9, 21, 50],  # Traditionele stock parameters
            'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70,
            'volatility_factor': 0.5, 'risk_per_trade': 0.01}


def get_risk_params(symbol: str) -> dict:
    """
    Haalt risico management parameters op per asset class.

    Parameters:
    -----------
    symbol : str
        Trading instrument symbool

    Returns:
    --------
    dict
        Dictionary met risico parameters voor FTMO compliance
    """
    asset_class = detect_asset_class(symbol)

    # Basis risico parameters (FTMO compliant)
    risk_params = {'max_daily_loss': 0.05,  # Maximaal 5% verlies per dag
        'max_total_loss': 0.10,  # Maximaal 10% totaal verlies
        'profit_target': 0.10,  # 10% winstdoel per cyclus
        'min_trading_days': 4,  # Minimaal 4 handelsdagen
    }

    # Asset-specifieke aanpassingen
    if asset_class == 'forex':
        risk_params['risk_per_trade'] = 0.01  # 1% risico
    elif asset_class == 'crypto':
        risk_params['risk_per_trade'] = 0.005  # 0.5% risico (voorzichtiger)
        risk_params['max_open_positions'] = 1  # Slechts 1 crypto positie tegelijk
    elif asset_class == 'index':
        risk_params['risk_per_trade'] = 0.01
        risk_params['max_open_positions'] = 2  # Maximaal 2 index posities
    else:  # stocks
        risk_params['risk_per_trade'] = 0.01
        risk_params['max_open_positions'] = 3  # Maximaal 3 stock posities

    return risk_params