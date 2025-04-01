# Sophy3 Trading Framework

Een gestroomlijnde, vectorized trading framework gebaseerd op VectorBT voor FTMO en andere prop trading firms.

## Kenmerken

- Geoptimaliseerd voor snelheid met vectorized berekeningen
- MT5 integratie voor backtesting en live trading
- Asset-class specifieke parameterisatie
- FTMO-compatibele risk management

## Installatie

```bash
# Clone de repository
git clone https://github.com/gebruikersnaam/sophy3.git
cd sophy3

# Installeer afhankelijkheden
pip install -r requirements.txt
```

## Gebruik

### Backtesting

```bash
python scripts/backtest.py --symbol EURUSD --timeframe H1 --capital 10000 --risk 0.01 --detailed
```

### Live Trading

```bash
python scripts/live_trade.py --account 123456 --password xxx --server MyBroker --symbols EURUSD GBPUSD --timeframe H1
```

## Structuur

- `strategies/`: Trading strategieën
- `data/`: Data toegang & voorbereiding
- `risk/`: Risicomanagement
- `trading/`: Live trading componenten
- `utils/`: Hulpfuncties
- `scripts/`: Command-line tools

## Licentie

MIT
