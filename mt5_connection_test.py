"""
MT5 Troubleshoot Script
Functie: Diagnosticeren en oplossen van MT5 verbindingsproblemen
Auteur: AI Trading Assistant
"""

import os
import time
import sys
import subprocess
from datetime import datetime

# Controleer of MT5 package is geïnstalleerd
try:
    import MetaTrader5 as mt5
    print("✓ MetaTrader5 package gevonden")
except ImportError:
    print("✗ MetaTrader5 package niet gevonden. Installeer met: pip install MetaTrader5")
    sys.exit(1)

# Controleer besturingssysteem en privileges
print(f"Besturingssysteem: {os.name}")
print(f"Python versie: {sys.version}")
print(f"MT5 package versie: {mt5.__version__ if hasattr(mt5, '__version__') else 'Onbekend'}")

# Controleer of MT5 terminal draait
def is_mt5_running():
    """Controleer of MT5 terminal draait via processen"""
    if os.name == 'nt':  # Windows
        try:
            output = subprocess.check_output('tasklist', shell=True).decode()
            return 'terminal64.exe' in output or 'terminal.exe' in output
        except:
            return "Onbekend"
    else:  # Mac/Linux
        return "Onbekend (niet-Windows OS)"

mt5_running = is_mt5_running()
print(f"MT5 terminal draait: {mt5_running}")

# Diagnostische info
print("\n=== MT5 Verbindingstests ===")
print("1. Test met direct initialize...")

# Probeer directe verbinding
try:
    start = time.time()
    initialized = mt5.initialize()
    elapsed = time.time() - start

    if initialized:
        print(f"✓ Verbinding gelukt in {elapsed:.2f} seconden")
        terminal_info = mt5.terminal_info()
        if terminal_info is not None:
            print(f"  • Terminal pad: {terminal_info.path}")
            print(f"  • Verbonden: {'Ja' if terminal_info.connected else 'Nee'}")
            print(f"  • CPU-cores: {terminal_info.cpu_cores}")
            print(f"  • Geheugen: {terminal_info.memory_total/1024/1024:.1f} MB")
        mt5.shutdown()
    else:
        error = mt5.last_error()
        print(f"✗ Verbinding mislukt: {error}")

        # Suggereer oplossingen
        if "timeout" in str(error).lower():
            print("\n=== Mogelijke oplossingen voor timeout ===")
            print("1. Zorg dat MT5 is opgestart en ingelogd")
            print("2. Herstart MT5 en zorg dat het volledig is geladen")
            print("3. Probeer MT5 als administrator te starten")
            print("4. Controleer firewall/antivirus instellingen")
            print("5. Kijk hieronder voor workaround met terminal_path...")
except Exception as e:
    print(f"✗ Test mislukt met error: {str(e)}")

# Test met terminal_path
print("\n2. Test met terminal_path parameter...")
try:
    # Standaard MT5 installatie paden proberen
    paths = [
        r"C:\Program Files\MetaTrader 5\terminal64.exe",
        r"C:\Program Files (x86)\MetaTrader 5\terminal.exe",
        # Voeg hier andere mogelijke paden toe
    ]

    # Vraag gebruiker om pad indien het niet in de lijst staat
    user_path = input("Voer het pad naar je MT5 terminal in (laat leeg voor standaard paden): ")
    if user_path:
        paths.insert(0, user_path)

    success = False
    for path in paths:
        if os.path.exists(path):
            print(f"Proberen pad: {path}")
            try:
                start = time.time()
                initialized = mt5.initialize(path=path)
                elapsed = time.time() - start

                if initialized:
                    print(f"✓ Verbinding gelukt met pad in {elapsed:.2f} seconden")
                    success = True

                    # Test data ophalen
                    print("\n3. Test data ophalen...")
                    try:
                        start = time.time()
                        rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 10)
                        elapsed = time.time() - start

                        if rates is not None and len(rates) > 0:
                            print(f"✓ Data opgehaald: {len(rates)} bars in {elapsed:.2f} seconden")
                            first_bar = datetime.fromtimestamp(rates[0][0])
                            print(f"  • Eerste bar: {first_bar}")
                        else:
                            print(f"✗ Geen data verkregen: {mt5.last_error()}")
                    except Exception as e:
                        print(f"✗ Data ophalen mislukt: {str(e)}")

                    mt5.shutdown()
                    break
                else:
                    print(f"✗ Verbinding mislukt met pad: {mt5.last_error()}")
            except Exception as e:
                print(f"✗ Error met pad {path}: {str(e)}")
        else:
            print(f"✗ Pad bestaat niet: {path}")

    if not success:
        print("\n✗ Kon geen verbinding maken met MT5 via beschikbare paden")

except Exception as e:
    print(f"✗ Terminal pad test mislukt: {str(e)}")

print("\n=== Aanbevelingen ===")
print("• Zorg dat MT5 is opgestart en volledig geladen voor je Python script start")
print("• Probeer het correcte pad naar terminal.exe/terminal64.exe te gebruiken")
print("• Probeer je Python script met administrator rechten te starten")
print("• Controleer of je firewall/antivirus Python niet blokkeert")
print("• Als niets werkt, overweeg om over te schakelen naar mock data voor testen")