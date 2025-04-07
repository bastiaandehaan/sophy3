"""
Sophy3 - Risk Manager
Functie: FTMO-compliant risk management voor prop trading
Auteur: AI Trading Assistant
Laatste update: 2025-04-06

Gebruik:
  Deze module bevat de risk management logica voor het trading systeem,
  inclusief position sizing en FTMO compliance monitoring.

Dependencies:
  - pandas
  - numpy
"""

import logging
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd

# Stel logger in
logger = logging.getLogger(__name__)

# FTMO compliance constants
FTMO_MAX_DAILY_LOSS_PCT = 0.05      # 5% max dagelijks verlies
FTMO_MAX_TOTAL_LOSS_PCT = 0.10      # 10% max totaal verlies
FTMO_PROFIT_TARGET_PCT = 0.10       # 10% winstdoel
FTMO_MIN_TRADING_DAYS = 10          # Minimum handelsdagen voor challenge
FTMO_MAX_POSITION_SIZE_PCT = 0.05   # 5% max positiegrootte
FTMO_MAX_DAILY_TRADES = 0           # 0 = onbeperkt

class FTMORiskManager:
    """FTMO-compliant risk manager voor prop trading."""

    def __init__(self, initial_capital: float = 10000.0,
                 max_daily_loss_pct: float = FTMO_MAX_DAILY_LOSS_PCT,
                 max_total_loss_pct: float = FTMO_MAX_TOTAL_LOSS_PCT,
                 profit_target_pct: float = FTMO_PROFIT_TARGET_PCT,
                 min_trading_days: int = FTMO_MIN_TRADING_DAYS,
                 max_position_size_pct: float = FTMO_MAX_POSITION_SIZE_PCT,
                 max_daily_trades: int = FTMO_MAX_DAILY_TRADES):
        """
        Initialiseert de risk manager met FTMO compliance parameters.

        Parameters:
        -----------
        initial_capital : float
            Startbedrag van het account
        max_daily_loss_pct : float
            Maximaal verlies per dag als percentage (bijv. 0.05 voor 5%)
        max_total_loss_pct : float
            Maximaal totaalverlies als percentage
        profit_target_pct : float
            Winstdoel als percentage
        min_trading_days : int
            Minimaal aantal handelsdagen voor winstopname
        max_position_size_pct : float
            Maximale positiegrootte als percentage van het account
        max_daily_trades : int
            Maximaal aantal trades per dag (0 = onbeperkt)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_daily_loss = initial_capital * max_daily_loss_pct
        self.max_total_loss = initial_capital * max_total_loss_pct
        self.profit_target = initial_capital * profit_target_pct
        self.min_trading_days = min_trading_days
        self.max_position_size = initial_capital * max_position_size_pct
        self.max_daily_trades = max_daily_trades

        # Tracking variabelen
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.trading_days = 0
        self.current_date = date.today()
        self.open_positions = {}  # symbol -> position_info dict
        self.daily_trades_count = 0

        # FTMO-specifieke counters
        self.consecutive_loss_days = 0
        self.profit_days = 0
        self.loss_days = 0
        self.daily_history = []

        # Risico adaptatie
        self.risk_factor = 1.0  # Dynamische risico aanpassing (1.0 = 100% van max risico)

        logger.info(f"FTMO Risk Manager geïnitialiseerd: kapitaal={initial_capital}, "
                    f"max dagelijks verlies={self.max_daily_loss}, "
                    f"max totaal verlies={self.max_total_loss}")

    def update_capital(self, new_capital: float):
        """Update het huidige kapitaal en herbereken limieten."""
        old_capital = self.current_capital
        self.current_capital = new_capital
        self.total_pnl = new_capital - self.initial_capital

        # Update limieten op basis van nieuw kapitaal
        if new_capital > self.initial_capital:
            # Verhoog alleen limieten als kapitaal is toegenomen
            self.max_daily_loss = new_capital * FTMO_MAX_DAILY_LOSS_PCT
            self.max_position_size = new_capital * FTMO_MAX_POSITION_SIZE_PCT

        logger.info(f"Kapitaal bijgewerkt: {old_capital:.2f} -> {new_capital:.2f}")
        logger.info(f"Nieuwe max dagelijks verlies: {self.max_daily_loss:.2f}")
        logger.info(f"Nieuwe max positiegrootte: {self.max_position_size:.2f}")

    def reset_daily_pnl(self):
        """Reset dagelijkse P&L tracking en verhoog trading days counter."""
        # Bewaar geschiedenis
        self.daily_history.append({
            'date': self.current_date,
            'pnl': self.daily_pnl,
            'trades': self.daily_trades_count,
            'ending_capital': self.current_capital
        })

        # Update consecutieve dagen tracking
        if self.daily_pnl < 0:
            self.consecutive_loss_days += 1
            self.loss_days += 1

            # Verlaag risicofactor na verliesdag
            self.risk_factor = max(0.5, self.risk_factor - 0.1)  # Min 50% van normaal risico
        else:
            self.consecutive_loss_days = 0
            self.profit_days += 1

            # Verhoog risicofactor na winstdag
            self.risk_factor = min(1.0, self.risk_factor + 0.05)  # Max 100% van normaal risico

        # Reset dagelijkse counters
        self.daily_pnl = 0.0
        self.daily_trades_count = 0
        self.current_date = date.today()
        self.trading_days += 1

        logger.info(f"Dagelijkse P&L gereset voor {self.current_date}")
        logger.info(f"Trading dagen: {self.trading_days}, Risicofactor: {self.risk_factor:.2f}")

    def check_new_day(self):
        """Controleert of er een nieuwe handelsdag is begonnen."""
        today = date.today()
        if today != self.current_date:
            logger.info(f"Nieuwe handelsdag: {self.current_date} -> {today}")
            self.reset_daily_pnl()
            return True
        return False

    def calculate_position_size(self, symbol: str, risk_per_trade: float,
                                entry_price: float, stop_loss_price: float,
                                lot_size: float = 100000) -> float:
        """
        Berekent positiegrootte op basis van risico per trade met FTMO compliance.

        Parameters:
        -----------
        symbol : str
            Instrument symbool
        risk_per_trade : float
            Risico per trade als percentage (bijv. 0.01 voor 1%)
        entry_price : float
            Entry prijs
        stop_loss_price : float
            Stop-loss niveau
        lot_size : float
            Standaard lot grootte (100000 voor forex)

        Returns:
        --------
        float
            Positiegrootte in lots
        """
        if entry_price <= stop_loss_price:
            logger.error(
                f"Entry price ({entry_price}) moet hoger zijn dan stop loss ({stop_loss_price})")
            return 0

        # Pas risico aan o.b.v. risicofactor (adaptieve risicobeheersing)
        adjusted_risk_per_trade = risk_per_trade * self.risk_factor

        # Maximaal risicobedrag voor deze trade
        risk_amount = self.current_capital * adjusted_risk_per_trade

        # FTMO Daily Loss Protection:
        # Als dagelijks verlies >50% van max, verlaag risico
        daily_loss_usage_pct = abs(self.daily_pnl) / self.max_daily_loss
        if daily_loss_usage_pct > 0.5:
            # Verlaag risico progressief naarmate we dichter bij limiet komen
            reduction_factor = 1.0 - daily_loss_usage_pct
            risk_amount *= reduction_factor
            logger.warning(f"Dagelijks verlies hoog ({daily_loss_usage_pct:.1%} van max), "
                          f"risico verlaagd met factor {reduction_factor:.2f}")

        # FTMO Total Loss Protection:
        # Als totaal verlies >50% van max, verlaag risico
        if self.total_pnl < 0:
            total_loss_usage_pct = abs(self.total_pnl) / self.max_total_loss
            if total_loss_usage_pct > 0.5:
                # Verlaag risico progressief naarmate we dichter bij limiet komen
                reduction_factor = 1.0 - total_loss_usage_pct
                risk_amount *= reduction_factor
                logger.warning(f"Totaal verlies hoog ({total_loss_usage_pct:.1%} van max), "
                              f"risico verlaagd met factor {reduction_factor:.2f}")

        # Begrens risicobedrag aan max dagelijks verlies-rest
        max_risk_today = self.max_daily_loss - abs(self.daily_pnl)
        risk_amount = min(risk_amount, max_risk_today)

        # Bereken lots gebaseerd op risico
        price_risk = abs(entry_price - stop_loss_price)
        units = risk_amount / price_risk
        lots = units / lot_size

        # Begrens positiegrootte (FTMO vereiste)
        position_value = entry_price * lot_size * lots
        if position_value > self.max_position_size:
            max_allowed_lots = self.max_position_size / (entry_price * lot_size)
            logger.warning(f"Positiegrootte begrensd: {lots:.2f} -> {max_allowed_lots:.2f} lots")
            lots = max_allowed_lots

        logger.info(f"Positiegrootte berekend voor {symbol}: {lots:.2f} lots "
                    f"(risicobedrag: {risk_amount:.2f}, prijsrisico: {price_risk:.5f})")

        return lots

    def can_open_position(self, symbol: str) -> bool:
        """
        Controleert of een nieuwe positie kan worden geopend met FTMO compliance.

        Parameters:
        -----------
        symbol : str
            Instrument symbool

        Returns:
        --------
        bool
            True als een nieuwe positie is toegestaan, anders False
        """
        # Controleer of symbool al open is
        if symbol in self.open_positions:
            logger.warning(f"Positie voor {symbol} is al open")
            return False

        # Controleer dagelijkse trade limiet
        if self.max_daily_trades > 0 and self.daily_trades_count >= self.max_daily_trades:
            logger.warning(f"Dagelijkse trade limiet bereikt: {self.daily_trades_count}")
            return False

        # Controleer of dagelijks verlies niet is bereikt
        if abs(self.daily_pnl) >= self.max_daily_loss:
            logger.warning(f"Dagelijks verlies bereikt: {self.daily_pnl:.2f}")
            return False

        # Controleer of totaal verlies niet is bereikt
        if self.total_pnl < 0 and abs(self.total_pnl) >= self.max_total_loss:
            logger.warning(f"Totaal verlies bereikt: {self.total_pnl:.2f}")
            return False

        # Controleer of we niet over veel consecutive loss dagen gaan
        if self.consecutive_loss_days >= 3 and self.daily_pnl < 0:
            # Bij 3+ consecutive loss dagen én verlies vandaag, handel conservatiever
            logger.warning(f"3+ consecutive loss dagen: {self.consecutive_loss_days}")
            return False

        return True

    def register_position(self, symbol: str, entry_price: float, stop_loss: float,
                          position_size: float):
        """
        Registreert een nieuwe open positie.

        Parameters:
        -----------
        symbol : str
            Instrument symbool
        entry_price : float
            Entry prijs
        stop_loss : float
            Stop-loss niveau
        position_size : float
            Positiegrootte in lots
        """
        if not self.can_open_position(symbol):
            logger.error(f"Kan positie voor {symbol} niet openen")
            return False

        # Registreer positie
        self.open_positions[symbol] = {
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'position_size': position_size,
            'entry_date': datetime.now(),
            'max_risk': abs(entry_price - stop_loss) * position_size,
            'symbol': symbol
        }

        # Verhoog dagelijkse trade teller
        self.daily_trades_count += 1

        logger.info(f"Positie geregistreerd voor {symbol}: "
                    f"prijs={entry_price}, stop={stop_loss}, grootte={position_size}")

        return True

    def update_position(self, symbol: str, current_price: float):
        """
        Werkt de status van een open positie bij.

        Parameters:
        -----------
        symbol : str
            Instrument symbool
        current_price : float
            Huidige marktprijs

        Returns:
        --------
        dict
            Positie informatie met P&L
        """
        if symbol not in self.open_positions:
            logger.warning(f"Geen open positie gevonden voor {symbol}")
            return None

        position = self.open_positions[symbol]

        # Bereken lopende P&L
        pip_pnl = current_price - position['entry_price']
        monetary_pnl = pip_pnl * position['position_size']

        # Update positie informatie
        position['current_price'] = current_price
        position['pip_pnl'] = pip_pnl
        position['monetary_pnl'] = monetary_pnl

        # Bereken huidige drawdown percentage
        if monetary_pnl < 0:
            dd_pct = abs(monetary_pnl) / self.initial_capital
            position['drawdown_pct'] = dd_pct
        else:
            position['drawdown_pct'] = 0.0

        return position

    def close_position(self, symbol: str, exit_price: float):
        """
        Sluit een open positie en registreert P&L.

        Parameters:
        -----------
        symbol : str
            Instrument symbool
        exit_price : float
            Exit prijs

        Returns:
        --------
        float
            Gerealiseerde P&L
        """
        if symbol not in self.open_positions:
            logger.warning(f"Geen open positie gevonden voor {symbol}")
            return 0

        position = self.open_positions[symbol]

        # Bereken P&L
        pip_pnl = exit_price - position['entry_price']
        monetary_pnl = pip_pnl * position['position_size']

        # Update P&L tracking
        self.daily_pnl += monetary_pnl
        self.total_pnl += monetary_pnl
        self.current_capital += monetary_pnl

        # Log resultaat
        logger.info(f"Positie gesloten voor {symbol}: entry={position['entry_price']}, "
                    f"exit={exit_price}, P&L={monetary_pnl:.2f}")

        # Verwijder positie uit open posities
        del self.open_positions[symbol]

        return monetary_pnl

    def should_exit_position(self, symbol: str, current_price: float) -> bool:
        """
        Controleert of een positie gesloten moet worden op basis van stop-loss.

        Parameters:
        -----------
        symbol : str
            Instrument symbool
        current_price : float
            Huidige marktprijs

        Returns:
        --------
        bool
            True als de positie gesloten moet worden
        """
        if symbol not in self.open_positions:
            return False

        position = self.open_positions[symbol]

        # Stop-loss check
        if current_price <= position['stop_loss']:
            logger.info(f"Stop-loss bereikt voor {symbol}: {position['stop_loss']}")
            return True

        # FTMO Daily Loss Protection
        # Als dagelijks verlies (inclusief deze positie) te dicht bij max komt
        position_pnl = (current_price - position['entry_price']) * position['position_size']
        projected_daily_pnl = self.daily_pnl + position_pnl

        if projected_daily_pnl < 0 and abs(projected_daily_pnl) > (self.max_daily_loss * 0.95):
            logger.warning(f"Dagelijks verlies limiet nadert, exit {symbol}")
            return True

        return False

    def should_take_profit(self) -> bool:
        """
        Controleert of het winstdoel is bereikt en voldoende handelsdagen zijn verstreken.

        Returns:
        --------
        bool
            True als winst genomen moet worden
        """
        if self.total_pnl >= self.profit_target and self.trading_days >= self.min_trading_days:
            logger.info(
                f"Winstdoel bereikt: {self.total_pnl:.2f} na {self.trading_days} dagen")
            return True

        return False

    def get_account_status(self) -> dict:
        """
        Geeft huidige account status.

        Returns:
        --------
        dict
            Account status informatie
        """
        # Bereken percentages
        daily_loss_pct = abs(self.daily_pnl) / self.initial_capital if self.daily_pnl < 0 else 0
        total_loss_pct = abs(self.total_pnl) / self.initial_capital if self.total_pnl < 0 else 0
        profit_pct = self.total_pnl / self.initial_capital if self.total_pnl > 0 else 0

        # Bereken FTMO compliance metrics
        daily_loss_threshold = daily_loss_pct / FTMO_MAX_DAILY_LOSS_PCT
        total_loss_threshold = total_loss_pct / FTMO_MAX_TOTAL_LOSS_PCT
        profit_target_threshold = profit_pct / FTMO_PROFIT_TARGET_PCT

        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'trading_days': self.trading_days,
            'max_daily_loss': self.max_daily_loss,
            'max_total_loss': self.max_total_loss,
            'daily_loss_pct': daily_loss_pct,
            'total_loss_pct': total_loss_pct,
            'profit_target': self.profit_target,
            'profit_pct': profit_pct,
            'open_positions': len(self.open_positions),
            'daily_trades_count': self.daily_trades_count,
            'consecutive_loss_days': self.consecutive_loss_days,
            'risk_factor': self.risk_factor,
            'ftmo_daily_loss_threshold': daily_loss_threshold,
            'ftmo_total_loss_threshold': total_loss_threshold,
            'ftmo_profit_target_threshold': profit_target_threshold,
            'current_date': self.current_date
        }

    def get_position_summary(self) -> pd.DataFrame:
        """
        Genereert een overzicht van alle open posities.

        Returns:
        --------
        pandas.DataFrame
            Overzicht van open posities met P&L informatie
        """
        if not self.open_positions:
            return pd.DataFrame()

        positions = []
        for symbol, pos in self.open_positions.items():
            pos_copy = pos.copy()
            pos_copy['symbol'] = symbol
            positions.append(pos_copy)

        df = pd.DataFrame(positions)

        # Bereken extra metrics
        if 'monetary_pnl' in df.columns:
            df['pnl_pct'] = df['monetary_pnl'] / self.current_capital

        return df