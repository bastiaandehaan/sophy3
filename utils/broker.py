import MetaTrader5 as mt5
import pandas as pd
import datetime
import pytz

def get_symbol_details():
    """Haal gedetailleerde informatie op over alle MT5-symbolen."""
    # Initialiseer verbinding met MT5
    if not mt5.initialize():
        print(f"MT5 initialisatie mislukt: {mt5.last_error()}")
        return

    print("Verbinding met MT5 succesvol gemaakt")

    try:
        # Haal alle symbolen op
        symbols = mt5.symbols_get()
        if not symbols:
            print(f"Kon geen symbolen ophalen: {mt5.last_error()}")
            return

        # Lijst voor symbolen-data
        symbols_data = []
        for sym in symbols:
            # Haal extra info op met symbol_info
            info = mt5.symbol_info(sym.name)
            if not info:
                continue

            # Converteer handelsuren naar leesbare tijden (in UTC)
            timezone = pytz.timezone('UTC')  # MT5 gebruikt UTC
            session_open = datetime.datetime.fromtimestamp(info.session_open, tz=timezone).strftime('%Y-%m-%d %H:%M:%S') if info.session_open else "N/A"
            session_close = datetime.datetime.fromtimestamp(info.session_close, tz=timezone).strftime('%Y-%m-%d %H:%M:%S') if info.session_close else "N/A"

            # Bereken spread in punten en converteer naar pips
            spread_points = info.spread
            pip_size = info.point * 10 if 'JPY' in sym.name else info.point  # Aanpassing voor JPY-paren
            spread_pips = spread_points * pip_size if pip_size else spread_points

            # Commissie (als beschikbaar)
            commission = info.trade_commission if hasattr(info, 'trade_commission') else 0.0

            # Data verzamelen
            symbol_dict = {
                'Naam': sym.name,
                'Beschrijving': sym.description,
                'Valutabasis': sym.currency_base,
                'Winstvaluta': sym.currency_profit,
                'Spread (pips)': round(spread_pips, 2),
                'Commissie': commission,
                'Lot Min': sym.volume_min,
                'Lot Max': sym.volume_max,
                'Lot Stap': sym.volume_step,
                'Punt': sym.point,
                'Pip Waarde': pip_size,
                'Contractgrootte': info.trade_contract_size,
                'Tick Grootte': info.tick_size,
                'Tick Waarde': info.tick_value,
                'Margin Valuta': sym.currency_margin,
                'Handelsmodus': info.trade_mode,  # 0=disabled, 1=long only, 2=short only, 3=full
                'Swap Long': info.swap_long,
                'Swap Short': info.swap_short,
                'Swap 3-daags': info.swap_rollover3days,
                'Sessiestart (UTC)': session_open,
                'Sessiesluit (UTC)': session_close,
                'Tijdstip Laatste Update': datetime.datetime.fromtimestamp(info.time, tz=timezone).strftime('%Y-%m-%d %H:%M:%S'),
                'Zichtbaar': sym.visible,
                'Pad': sym.path,
                'Marktsector': sym.path.split('\\')[0] if '\\' in sym.path else 'Overig'
            }
            symbols_data.append(symbol_dict)

        # Converteer naar DataFrame
        symbols_df = pd.DataFrame(symbols_data)
        symbols_df.sort_values('Naam', inplace=True)

        # Console-output
        print(f"\nTotaal aantal beschikbare symbolen: {len(symbols)}")
        print("\nVOORBEELD VAN BESCHIKBARE SYMBOLEN (eerste 20):")
        print(symbols_df.head(20).to_string(index=False))

        # Opslaan naar CSV
        csv_file = 'mt5_symbol_details.csv'
        symbols_df.to_csv(csv_file, index=False)
        print(f"\nAlle symboolinformatie opgeslagen in '{csv_file}'")

        # Analyse per marktsector
        print("\nAANTAL SYMBOLEN PER MARKTSECTOR:")
        sector_counts = symbols_df['Marktsector'].value_counts()
        for sector, count in sector_counts.items():
            print(f"{sector}: {count}")

        # Specifieke analyse: Forex-paren
        forex_df = symbols_df[symbols_df['Marktsector'].str.contains('forex|FX', case=False, na=False)]
        if not forex_df.empty:
            print("\nFOREX-PAREN (eerste 10):")
            print(forex_df[['Naam', 'Spread (pips)', 'Commissie', 'Lot Min', 'Swap Long', 'Swap Short']].head(10).to_string(index=False))

        # Specifieke analyse: Indices (bijv. DAX)
        indices_df = symbols_df[symbols_df['Naam'].str.contains('DAX|SPX|NDX|DJI|FTSE|NKY', case=False, na=False)]
        if not indices_df.empty:
            print("\nINDEX-INSTRUMENTEN:")
            print(indices_df[['Naam', 'Spread (pips)', 'Commissie', 'Sessiestart (UTC)', 'Sessiesluit (UTC)']].to_string(index=False))

        # Statistische samenvatting
        print("\nSTATISTISCHE SAMENVATTING:")
        stats = {
            'Gemiddelde Spread (pips)': symbols_df['Spread (pips)'].mean(),
            'Gemiddelde Commissie': symbols_df['Commissie'].mean(),
            'Min Lot Grootte': symbols_df['Lot Min'].min(),
            'Max Lot Grootte': symbols_df['Lot Max'].max()
        }
        for stat, value in stats.items():
            print(f"{stat}: {value:.2f}")

    except Exception as e:
        print(f"Fout tijdens uitvoering: {e}")
    finally:
        mt5.shutdown()
        print("\nMT5 verbinding gesloten")


if __name__ == "__main__":
    get_symbol_details()