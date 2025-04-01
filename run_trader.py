"""
Sophy3 - Run Trader
Functie: Eenvoudige CLI-interface voor het uitvoeren van verschillende backtests en analyses
Auteur: AI Trading Assistant
Laatste update: 2025-04-01

Gebruik:
  python run_trader.py

Dependencies:
  - argparse
  - subprocess
"""

import os
import time
import subprocess
import argparse
from datetime import datetime


def print_banner():
    """Toon een welkomstbanner voor Sophy3."""
    print("=" * 80)
    print("""
    ███████╗ ██████╗ ██████╗ ██╗  ██╗██╗   ██╗██████╗ 
    ██╔════╝██╔═══██╗██╔══██╗██║  ██║╚██╗ ██╔╝╚════██╗
    ███████╗██║   ██║██████╔╝███████║ ╚████╔╝  █████╔╝
    ╚════██║██║   ██║██╔═══╝ ██╔══██║  ╚██╔╝  ██╔═══╝ 
    ███████║╚██████╔╝██║     ██║  ██║   ██║   ███████╗
    ╚══════╝ ╚═════╝ ╚═╝     ╚═╝  ╚═╝   ╚═╝   ╚══════╝

    Vectorized Trading Framework v3.0
    """)
    print("=" * 80)
    print(f"Huidige tijd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


def print_menu():
    """Toon het hoofdmenu."""
    print("\nSophy3 Trading Framework - Hoofdmenu")
    print("=" * 50)
    print("1. Quick Backtest")
    print("2. Gedetailleerde Backtest met Visualisatie")
    print("3. Parameter Optimalisatie")
    print("4. Asset Class Vergelijking")
    print("5. Specifieke Commando Uitvoeren")
    print("6. Installeer Vereiste Packages")
    print("0. Afsluiten")
    print("=" * 50)


def print_asset_menu():
    """Toon het menu voor asset selectie."""
    print("\nSelecteer Asset Class / Instrument:")
    print("=" * 50)
    print("1. Forex - EURUSD (H1)")
    print("2. Forex - GBPUSD (H1)")
    print("3. Crypto - BTCUSD (H4)")
    print("4. Indices - SPX (D1)")
    print("5. Stocks - AAPL (D1)")
    print("6. Ander instrument (handmatig invoeren)")
    print("=" * 50)


def print_optimization_menu():
    """Toon het menu voor optimalisatie."""
    print("\nWat wil je optimaliseren?")
    print("=" * 50)
    print("1. EMA Periodes")
    print("2. RSI Parameters")
    print("3. Volatiliteitsfilter")
    print("=" * 50)


def print_metrics_menu():
    """Toon het menu voor optimalisatiemetrics."""
    print("\nSelecteer Optimalisatie Metric:")
    print("=" * 50)
    print("1. Sharpe Ratio (risk-adjusted returns)")
    print("2. Sortino Ratio (downside risk focus)")
    print("3. Max Drawdown (minimaliseren)")
    print("4. Profit Factor (winsten / verliezen)")
    print("5. Win Rate (percentage winnende trades)")
    print("=" * 50)


def run_command(command, verbose=True):
    """
    Voer een commando uit en toon real-time output.

    Parameters:
    -----------
    command : str
        Het uit te voeren commando
    verbose : bool
        Indien True, toon extra informatie en timing
    """
    if verbose:
        print("\n" + "=" * 80)
        print(f"Uitvoeren commando: {command}")
        print("=" * 80)
        start_time = time.time()

    # Voer het commando direct uit, zonder output capture
    # Dit is veiliger op Windows en vermijdt Unicode-problemen
    process = subprocess.call(command, shell=True)

    if verbose:
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 80)
        print(f"Commando voltooid in {elapsed_time:.2f} seconden")
        print("=" * 80)

    return process


def select_asset():
    """Selecteer een asset en timeframe."""
    print_asset_menu()
    choice = input("Keuze (1-6): ")

    assets = {"1": {"symbol": "EURUSD", "timeframe": "H1", "preset": "forex"},
        "2": {"symbol": "GBPUSD", "timeframe": "H1", "preset": "forex"},
        "3": {"symbol": "BTCUSD", "timeframe": "H4", "preset": "crypto"},
        "4": {"symbol": "SPX", "timeframe": "D1", "preset": "index"},
        "5": {"symbol": "AAPL", "timeframe": "D1", "preset": "stock"}}

    if choice in assets:
        return assets[choice]
    elif choice == "6":
        symbol = input("Voer symbool in (bijv. EURUSD): ")
        timeframe = input("Voer timeframe in (M1, M5, M15, H1, H4, D1): ")
        preset = input("Voer asset class in (forex, crypto, index, stock): ")
        return {"symbol": symbol, "timeframe": timeframe, "preset": preset}
    else:
        print("Ongeldige keuze, standaard EURUSD H1 wordt gebruikt.")
        return assets["1"]


def select_optimization_param():
    """Selecteer parameter voor optimalisatie."""
    print_optimization_menu()
    choice = input("Keuze (1-3): ")

    params = {"1": "ema", "2": "rsi", "3": "volatility"}

    if choice in params:
        return params[choice]
    else:
        print("Ongeldige keuze, standaard EMA wordt gebruikt.")
        return params["1"]


def select_optimization_metric():
    """Selecteer metric voor optimalisatie."""
    print_metrics_menu()
    choice = input("Keuze (1-5): ")

    metrics = {"1": "sharpe", "2": "sortino", "3": "max_drawdown", "4": "profit_factor",
        "5": "win_rate"}

    if choice in metrics:
        return metrics[choice]
    else:
        print("Ongeldige keuze, standaard Sharpe ratio wordt gebruikt.")
        return metrics["1"]


def quick_backtest():
    """Voer een snelle backtest uit."""
    asset = select_asset()

    command = (f"python scripts/backtest.py "
               f"--symbol {asset['symbol']} "
               f"--timeframe {asset['timeframe']} "
               f"--parameter-preset {asset['preset']} "
               f"--capital 10000 --risk 0.01")

    run_command(command)


def detailed_backtest():
    """Voer een gedetailleerde backtest uit met visualisatie."""
    asset = select_asset()

    command = (f"python scripts/backtest.py "
               f"--symbol {asset['symbol']} "
               f"--timeframe {asset['timeframe']} "
               f"--parameter-preset {asset['preset']} "
               f"--capital 10000 --risk 0.01 "
               f"--detailed --plot")

    run_command(command)


def parameter_optimization():
    """Voer parameteroptimalisatie uit."""
    asset = select_asset()
    param = select_optimization_param()
    metric = select_optimization_metric()

    command = (f"python scripts/backtest.py "
               f"--symbol {asset['symbol']} "
               f"--timeframe {asset['timeframe']} "
               f"--parameter-preset {asset['preset']} "
               f"--capital 10000 --risk 0.01 "
               f"--optimize --optimize-param {param} "
               f"--optimize-metric {metric} "
               f"--detailed")

    run_command(command)


def asset_class_comparison():
    """Vergelijk verschillende asset classes."""
    print("\nVergelijking van verschillende asset classes...")

    assets = [{"symbol": "EURUSD", "timeframe": "H1", "preset": "forex"},
        {"symbol": "BTCUSD", "timeframe": "H4", "preset": "crypto"},
        {"symbol": "SPX", "timeframe": "D1", "preset": "index"},
        {"symbol": "AAPL", "timeframe": "D1", "preset": "stock"}]

    results = []

    for asset in assets:
        print(f"\nTesten van {asset['symbol']} ({asset['timeframe']})...")

        command = (f"python scripts/backtest.py "
                   f"--symbol {asset['symbol']} "
                   f"--timeframe {asset['timeframe']} "
                   f"--parameter-preset {asset['preset']} "
                   f"--capital 10000 --risk 0.01")

        run_command(command, verbose=False)

        # Hier zou je de resultaten kunnen opslaan en vergelijken
        # Dit is een vereenvoudigde implementatie die alleen de commando's uitvoert

    print("\nVergelijking voltooid. Bekijk de logs voor gedetailleerde resultaten.")


def custom_command():
    """Voer een aangepast commando uit."""
    print("\nVoer een aangepast backtest commando in.")
    print("Voorbeeld: --symbol EURUSD --timeframe H1 --detailed --plot")

    command_args = input("\nCommando argumenten: ")
    full_command = f"python scripts/backtest.py {command_args}"

    run_command(full_command)


def install_requirements():
    """Installeer vereiste packages."""
    print("\nInstalleren van vereiste packages...")

    packages = ["vectorbt", "pandas", "numpy", "matplotlib", "MetaTrader5", "tqdm"]

    for package in packages:
        command = f"pip install {package}"
        print(f"\nInstalleren van {package}...")
        run_command(command)

    print("\nInstallatie voltooid.")


def main():
    """Hoofdfunctie voor het CLI-menu."""
    print_banner()

    while True:
        print_menu()
        choice = input("Keuze (0-6): ")

        if choice == "0":
            print(
                "\nBedankt voor het gebruiken van Sophy3 Trading Framework. Tot ziens!")
            break
        elif choice == "1":
            quick_backtest()
        elif choice == "2":
            detailed_backtest()
        elif choice == "3":
            parameter_optimization()
        elif choice == "4":
            asset_class_comparison()
        elif choice == "5":
            custom_command()
        elif choice == "6":
            install_requirements()
        else:
            print("Ongeldige keuze, probeer opnieuw.")


if __name__ == "__main__":
    main()