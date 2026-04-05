import pandas as pd

data = pd.read_csv("Sample7_data.csv")
print(data)
print(data.columns)
print(data['product-category'].unique())
print(data.describe())