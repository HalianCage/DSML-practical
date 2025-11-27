import pandas as pd

df = pd.read_csv("./Datasets/Telecom Churn.csv")

print("=======Printing first few lines==========")
print(df.head())

print("==========printing max values===========")
print(df.max(numeric_only=True))

numeric_df = df.select_dtypes(include=[int, float])

print("=================Quartiles============")
print(numeric_df.quantile([0.25, 0.5, 0.75]))