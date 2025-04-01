import pandas as pd

# Pad naar je CSV-bestand
csv_path = r"C:\Users\Gebruiker\PycharmProjects\HIST_DATA\EURUSD_H1_combined.csv"

# Lees het CSV-bestand
df = pd.read_csv(csv_path)

# Hernoem de kolom 'datetime' naar 'time'
df.rename(columns={'datetime': 'time'}, inplace=True)

# Sla het aangepaste bestand op
df.to_csv(csv_path, index=False)

print("Kolom hernoemd en bestand opgeslagen.")