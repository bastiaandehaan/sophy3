"""
Sophy3 - Risk Manager
Functie: FTMO-compliant risk management voor prop trading
Auteur: AI Trading Assistant
Laatste update: 2025-04-01

Gebruik:
  Deze module bevat de risk management logica voor het trading systeem,
  inclusief position sizing en FTMO compliance monitoring.

Dependencies:
  - pandas
  - numpy
"""

import logging
from datetime import datetime, date

# Stel logger in
logger = logging.getLogger(__name__)


class FTMORiskManager:
    """Risk manager voor FTMO compliance."""

    def __init__(self, initial_capital: float = 10000.0,
                 max_daily_loss_pct: float = 0.05, max_total_loss_pct: float = 0.10,
                 profit_target_pct: float = 0.10, min_trading_days: int = 4):
        """
        Initialiseert de risk manager.

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
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_daily_loss = initial_capital * max_daily_loss_pct
        self.max_total_loss = initial_capital * max_total_loss_pct
        self.profit_target = initial_capital * profit_target_pct
        self.min_trading_days = min_trading_days

        # Tracking variabelen
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.trading_days = 0
        self.current_date = date.today()
        self.open_positions = {}  # symbol -> position_info dict

        logger.info(f"Risk Manager geïnitialiseerd: kapitaal={initial_capital}, "
                    f"max dagelijks verlies={self.max_daily_loss}, "
                    f"max totaal verlies={self.max_total_loss}")

    def update_capital(self, new_capital: float):
        """Update het huidige kapitaal."""
        old_capital = self.current_capital
        self.current_capital = new_capital
        self.total_pnl = new_capital - self.initial_capital
        logger.info(f"Kapitaal bijgewerkt: {old_capital} -> {new_capital}")

    def reset_daily_pnl(self):
        """Reset dagelijkse P&L tracking."""
        self.daily_pnl = 0.0
        self.current_date = date.today()
        logger.info(f"Dagelijkse P&L gereset voor {self.current_date}")

    def check_new_day(self):
        """Controleert of er een nieuwe handelsdag is begonnen."""
        today = date.today()
        if today != self.current_date:
            logger.info(f"Nieuwe handelsdag: {self.current_date} -> {today}")
            self.trading_days += 1
            self.reset_daily_pnl()
            return True
        return False

    def calculate_position_size(self, symbol: str, risk_per_trade: float,
                                entry_price: float, stop_loss_price: float,
                                lot_size: float = 100000) -> float:
        """
        Berekent positiegrootte op basis van risico per trade.

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

        # Maximaal risicobedrag voor deze trade
        risk_amount = self.current_capital * risk_per_trade

        # Controleer of dagelijks verlies niet wordt overschreden
        risk_amount = min(risk_amount, self.max_daily_loss - abs(self.daily_pnl))

        # Controleer of totaal verlies niet wordt overschreden
        risk_amount = min(risk_amount,
                          self.max_total_loss - abs(min(0, self.total_pnl)))

        # Bereken prijs per pip risico
        price_risk = abs(entry_price - stop_loss_price)

        # Bereken units (basisvaluta)
        units = risk_amount / price_risk

        # Converteer naar lots (forex standaard: 1 lot = 100,000 units)
        lots = units / lot_size

        logger.info(f"Positiegrootte berekend voor {symbol}: {lots:.2f} lots "
                    f"(risicobedrag: {risk_amount:.2f}, prijsrisico: {price_risk:.5f})")

        return lots

    def can_open_position(self, symbol: str) -> bool:
        """
        Controleert of een nieuwe positie kan worden geopend.

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

        # Controleer of dagelijks verlies niet is bereikt
        if abs(self.daily_pnl) >= self.max_daily_loss:
            logger.warning(f"Dagelijks verlies bereikt: {self.daily_pnl:.2f}")
            return False

        # Controleer of totaal verlies niet is bereikt
        if self.total_pnl < 0 and abs(self.total_pnl) >= self.max_total_loss:
            logger.warning(f"Totaal verlies bereikt: {self.total_pnl:.2f}")
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
        self.open_positions[symbol] = {'entry_price': entry_price,
            'stop_loss': stop_loss, 'position_size': position_size,
            'entry_date': datetime.now(),
            'max_risk': abs(entry_price - stop_loss) * position_size}

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
        return {'initial_capital': self.initial_capital,
            'current_capital': self.current_capital, 'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl, 'trading_days': self.trading_days,
            'max_daily_loss': self.max_daily_loss,
            'max_total_loss': self.max_total_loss, 'daily_loss_pct': abs(
                self.daily_pnl) / self.initial_capital if self.daily_pnl < 0 else 0,
            'total_loss_pct': abs(
                self.total_pnl) / self.initial_capital if self.total_pnl < 0 else 0,
            'profit_target': self.profit_target,
            'profit_target_pct': self.total_pnl / self.initial_capital if self.total_pnl > 0 else 0,
            'open_positions': len(self.open_positions),
            'current_date': self.current_date}