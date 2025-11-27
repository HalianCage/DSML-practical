# practice_dummy_dataset.py

import pandas as pd

dataset_path = "./Datasets/Titanic.csv"

# ------------------------------------------------------
# 1. CREATE A DUMMY TITANIC-LIKE DATASET
# ------------------------------------------------------

data = pd.read_csv(dataset_path)

df = pd.DataFrame(data)

# ------------------------------------------------------
# 2. SAVE DATA IN CSV AND XLSX FORMATS
# ------------------------------------------------------
df.to_csv("dummy_titanic.csv", index=False)
df.to_excel("dummy_titanic.xlsx", index=False)

print("Dummy dataset saved as dummy_titanic.csv and dummy_titanic.xlsx\n")

# ------------------------------------------------------
# 3. READ DATA FROM CSV & XLSX
# ------------------------------------------------------
df_csv = pd.read_csv("dummy_titanic.csv")
df_excel = pd.read_excel("dummy_titanic.xlsx")

print("=== DATA FROM CSV ===")
print(df_csv, "\n")

print("=== DATA FROM EXCEL ===")
print(df_excel, "\n")

# ------------------------------------------------------
# 4. INDEXING & SELECTING DATA
# ------------------------------------------------------

print("=== Selecting Columns (Name, Age, Sex) ===")
print(df_csv[["Name", "Age", "Sex"]], "\n")

print("=== Selecting Rows 0 to 2 using iloc ===")
print(df_csv.iloc[0:3], "\n")

print("=== Selecting by Condition (Age > 30) ===")
print(df_csv[df_csv["Age"] > 30], "\n")

# ------------------------------------------------------
# 5. SORTING DATA
# ------------------------------------------------------

print("=== Sort by Age (ascending) ===")
print(df_csv.sort_values("Age"), "\n")

print("=== Sort by Fare (descending) ===")
print(df_csv.sort_values("Fare", ascending=False), "\n")

# ------------------------------------------------------
# 6. DESCRIBE ATTRIBUTES
# ------------------------------------------------------

print("=== Describe Numeric Data ===")
print(df_csv.describe(), "\n")

print("=== Describe All Columns ===")
print(df_csv.describe(include="all"), "\n")

# ------------------------------------------------------
# 7. CHECKING DATA TYPES
# ------------------------------------------------------

print("=== Data Types of Each Column ===")
print(df_csv.dtypes, "\n")

print("=== Dataset Info ===")
df_csv.info()
