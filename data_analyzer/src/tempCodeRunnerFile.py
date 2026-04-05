import pandas as pd
from visualizer import data_prep,call_churn

data = pd.read_csv("Train_data.csv")
data['order-date'] = pd.to_datetime(data['order-date'])
final_data = data_prep(data)
churned_data = call_churn(final_data)
print(churned_data)