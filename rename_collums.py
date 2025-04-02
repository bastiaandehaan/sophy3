import os
import pandas as pd

# Pad naar je CSV bestanden
csv_dir = r"C:\Users\Gebruiker\PycharmProjects\HIST_DATA"

# Lijst alle CSV bestanden
csv_files = [f for f in os.listdir(csv_dir) if
             f.endswith('.csv') and f != "EURUSD_H1.csv"]

for file in csv_files:
    csv_path = os.path.join(csv_dir, file)
    print(f"Verwerken van: {file}")

    # Lees het CSV-bestand
    df = pd.read_csv(csv_path)

    # Hernoem de kolom 'datetime' naar 'time' als deze bestaat
    if 'datetime' in df.columns:
        df.rename(columns={'datetime': 'time'}, inplace=True)
        # Sla het aangepaste bestand op
        df.to_csv(csv_path, index=False)
        print(f"✓ Kolom hernoemd in {file}")
    else:
        print(f"× Kolom 'datetime' niet gevonden in {file}")

print("Alle bestanden verwerkt.")