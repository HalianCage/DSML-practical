import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("./Datasets/House Data.csv")

# ---- CLEANING HELPERS ----
def clean_price(x):
    if isinstance(x, str):
        x = x.replace("TL", "").replace(",", "")
        return float(x)
    return np.nan

def clean_square_meters(x):
    if isinstance(x, str):
        return float(x.replace("m2", "").strip())
    return np.nan

# ---- CLEAN RELEVANT NUMERIC COLUMNS ----
numeric_columns = [
    "price", "GrossSquareMeters", "NetSquareMeters",
    "NumberFloorsofBuilding", "NumberOfBathrooms", "NumberOfWCs",
    "NumberOfRooms", "FloorLocation",
    "RentalIncome", "NumberOfBalconies",
    "HallSquareMeters", "WCSquareMeters",
    "BathroomSquareMeters", "BalconySquareMeters"
]

# Convert price
df["price"] = df["price"].apply(clean_price)

# Convert m2 columns
df["GrossSquareMeters"] = df["GrossSquareMeters"].apply(clean_square_meters)
df["NetSquareMeters"] = df["NetSquareMeters"].apply(clean_square_meters)

# Convert numeric-like columns if possible
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# --------------------------------------------------------
# 🔹 1. COMPUTE STD, VARIANCE, PERCENTILES FOR EACH FEATURE
# --------------------------------------------------------

print("\n===== STANDARD DEVIATION =====")
print(df[numeric_columns].std())

print("\n===== VARIANCE =====")
print(df[numeric_columns].var())

print("\n===== PERCENTILES (25th, 50th, 75th) =====")
print(df[numeric_columns].quantile([0.25, 0.5, 0.75]))

# --------------------------------------------------------
# 🔹 2. HISTOGRAM FOR EACH NUMERIC FEATURE
# --------------------------------------------------------

df[numeric_columns].hist(figsize=(15, 15), bins=10)
plt.tight_layout()
plt.show()
