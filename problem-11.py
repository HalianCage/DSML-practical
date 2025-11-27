import pandas as pd

df = pd.read_csv('./Datasets/IRIS.csv')

# list down the features and their datatypes
print(df.dtypes)

# create a histogram for each feature
df.hist()
import matplotlib.pyplot as plt
plt.show()