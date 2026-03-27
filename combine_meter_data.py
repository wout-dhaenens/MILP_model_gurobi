import pandas as pd
import os

BASE = os.path.dirname(os.path.abspath(__file__))

# ── File paths ────────────────────────────────────────────────────────────────
FILES = {
    "afname_2B":   os.path.join(BASE, "20260324_1111_meetgegevens_2EANs_kwartieruurtotalen",
                                "Afname_Elektriciteit_541448860007782552.csv"),
    "injectie_2B": os.path.join(BASE, "20260324_1113_meetgegevens_3EANs_kwartieruurtotalen",
                                "Injectie_Elektriciteit_541448860008451808.csv"),
    "nr4":         os.path.join(BASE, "20260324_1111_meetgegevens_2EANs_kwartieruurtotalen",
                                "Elektriciteit_541448860015424703.csv"),
}

# ── Helper: load a file and return only kWh rows with a timestamp ─────────────
def load_kwh(path, registers):
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", decimal=",",
                     dtype={"Volume": str})
    df.columns = df.columns.str.strip()

    # Build timestamp from start interval
    df["timestamp"] = pd.to_datetime(
        df["Van (datum)"] + " " + df["Van (tijdstip)"],
        format="%d-%m-%Y %H:%M:%S"
    )

    # Keep only the requested registers and kWh unit
    df = df[df["Register"].isin(registers) & (df["Eenheid"] == "kWh")].copy()

    # Convert volume (comma decimal, empty = 0)
    df["Volume"] = (df["Volume"].str.replace(",", ".", regex=False)
                                .replace("", "0")
                                .fillna("0")
                                .astype(float))

    return df[["timestamp", "Register", "Volume"]]


# ── Load each source ──────────────────────────────────────────────────────────
print("Loading Afname 2B ...")
afname_2b = load_kwh(FILES["afname_2B"], ["Afname Actief"])
afname_2b = afname_2b.groupby("timestamp")["Volume"].sum().rename("Afname_2B_kWh")

print("Loading Injectie 2B ...")
injectie_2b = load_kwh(FILES["injectie_2B"], ["Injectie Actief"])
injectie_2b = injectie_2b.groupby("timestamp")["Volume"].sum().rename("Injectie_2B_kWh")

print("Loading nr4 (Dag + Nacht) ...")
afname_4 = load_kwh(FILES["nr4"], ["Afname Dag", "Afname Nacht"])
afname_4 = afname_4.groupby("timestamp")["Volume"].sum().rename("Afname_4_kWh")

injectie_4 = load_kwh(FILES["nr4"], ["Injectie Dag", "Injectie Nacht"])
injectie_4 = injectie_4.groupby("timestamp")["Volume"].sum().rename("Injectie_4_kWh")

# ── Combine into one DataFrame ────────────────────────────────────────────────
print("Combining ...")
df = pd.concat([afname_2b, injectie_2b, afname_4, injectie_4], axis=1).sort_index()
df.index.name = "timestamp"

# Total offtake and injection
df["Afname_totaal_kWh"]   = df["Afname_2B_kWh"].fillna(0)   + df["Afname_4_kWh"].fillna(0)
df["Injectie_totaal_kWh"] = df["Injectie_2B_kWh"].fillna(0) + df["Injectie_4_kWh"].fillna(0)

# ── Save ──────────────────────────────────────────────────────────────────────
out = os.path.join(BASE, "combined_grid_data.csv")
df.to_csv(out, sep=";", decimal=",", float_format="%.4f")
print(f"\nSaved to: {out}")
print(f"Rows: {len(df):,}  |  Period: {df.index[0]} to {df.index[-1]}")
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values per column:")
print(df.isnull().sum())
