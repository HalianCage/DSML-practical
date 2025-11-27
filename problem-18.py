import pandas as pd

# Load the dataset
df = pd.read_csv("./Datasets/House_Price.csv")

# Identify categorical and quantitative variables
categorical_vars = df.select_dtypes(include=['object', 'category']).columns
quantitative_vars = df.select_dtypes(include=['int64', 'float64']).columns

print("Categorical Variables:", list(categorical_vars))
print("Quantitative Variables:", list(quantitative_vars))

# Example: Choose any categorical column and any quantitative column
cat_col = "Neighborhood"      # Example categorical variable
quant_col = "SalePrice"       # Example quantitative variable

# Group summary statistics
grouped_summary = df.groupby(cat_col)[quant_col].agg(
    ["mean", "median", "min", "max", "std", "count"]
)

print("\nSummary Statistics of", quant_col, "grouped by", cat_col)
print(grouped_summary)



# General function for any variable
# def summary_by_category(df, categorical_col, quantitative_col):
#     return df.groupby(categorical_col)[quantitative_col].agg(
#         ["mean", "median", "min", "max", "std", "count"]
#     )

# # Example usage
# result = summary_by_category(df, "HouseStyle", "GrLivArea")
# print(result)