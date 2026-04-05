import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
import pyarrow as pa
from sklearn.linear_model import LogisticRegression 
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("Train_data.csv")
print(data)
data['order-date'] = pd.to_datetime(data['order-date'])

# 2. Find the most recent date in the entire dataset to act as "today"
ref_date = data['order-date'].max()
churned_df = data.groupby('customer-email').agg({
    'order-date': lambda x: (ref_date - x.max()).days,
    'order-id':'nunique',
    'total-value':'sum'
}).reset_index()


churned_df.columns = ['customer-email', 'Recency', 'Frequency', 'Total_Value']

churned_df['Churn_target'] = (churned_df['Frequency']==1).astype('int')
churned_df['Churn-Label'] = churned_df['Churn_target'].map({1:'Yes',0:'No'})

print("Class Counts:")
print(churned_df['Churn_target'].value_counts())
print(churned_df['Frequency'].describe())
column = ['Recency', 'Total_Value']


X = churned_df[column]
Y = churned_df['Churn_target']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train,X_test,Y_train,Y_test = train_test_split(X_scaled,Y,test_size=0.2,random_state=42,stratify=Y)
model = LogisticRegression()
model.fit(X_train,Y_train)
prediction = model.predict(X_test)

accuracy = accuracy_score(prediction,Y_test)

print(accuracy)
with open('churn_predictor1.pkl','wb') as f:
    pickle.dump(model,f)






