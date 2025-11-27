import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# -----------------------------
# 1. BOX PLOTS FOR EACH FEATURE
# -----------------------------
plt.figure(figsize=(10, 6))
df.boxplot()
plt.title("Box Plot of Iris Dataset Features")
plt.xlabel("Features")
plt.ylabel("Measurement (cm)")
plt.show()

# -----------------------------
# 2. OUTLIER DETECTION USING IQR
# -----------------------------
print("---- OUTLIER ANALYSIS USING IQR ----\n")

def detect_outliers(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    
    print(f"{column}:")
    print(f"  Q1 = {Q1:.2f}, Q3 = {Q3:.2f}, IQR = {IQR:.2f}")
    print(f"  Lower Bound = {lower_bound:.2f}, Upper Bound = {upper_bound:.2f}")
    
    if outliers.empty:
        print("  Outliers: None\n")
    else:
        print(f"  Outliers ({len(outliers)} values):")
        print(outliers.values, "\n")

# Apply to all numeric features
for col in df.columns:
    detect_outliers(col)
