import pickle
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


# path code
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


model    = pickle.load(open(os.path.join(BASE_DIR, 'churn_predictor.pkl'), 'rb'))
features = pickle.load(open(os.path.join(BASE_DIR, 'churn_features.pkl'), 'rb'))

df = pd.read_csv(os.path.join(BASE_DIR, "Sample_data1.csv"))
df['order-date'] = pd.to_datetime(df['order-date'])




customer_status = df.groupby('customer-email').agg(
        last_purchase = ('order-date','max'),
        total_orders = ('order-id','count'),
        total_revenue = ('total-value','sum'),
        average_order_value = ('total-value','mean'),
        return_count = ('status',
                        lambda x:(x=='Returned').sum()),
        cancel_count = ('status',
                        lambda x:(x=='Cancelled').sum())

    ).reset_index()

max_date = df['order-date'].max()

    #defining customer
customer_status['recency'] = (
    max_date - customer_status['last_purchase']
).dt.days

# churn definition
recency_threshold = customer_status['recency'].quantile(0.75)

customer_status['return_rate'] = (
    customer_status['return_count']/
    customer_status['total_orders'].clip(lower=1) # means if any value less than 1 it will be counted as 0
)

customer_status['cancelled_rate'] = (
    customer_status['cancel_count']/
    customer_status['total_orders'].clip(lower=1)
)

customer_first = df.groupby('customer-email')['order-date'].min().reset_index()
customer_first.columns = ['customer-email','first-purchase']

customer_status = customer_status.merge(
    customer_first,on='customer-email'
)
customer_status['customer-age-days'] = (
    df['order-date'].max()-customer_status['first-purchase']
).dt.days

# Purchase Frequency
customer_status['purchase-frequency'] = (
    customer_status['total_orders']/customer_status['customer-age-days'].clip(lower=1)
)
features = [
    'total_orders',
    'total_revenue',
    'average_order_value',     
    'return_rate',
    'cancelled_rate',          
    'customer-age-days',       
    'purchase-frequency'       
]   

print(customer_status)
print(customer_status.size)
print(customer_status.describe())

# =========== PREDICTION PART STARTS HERE =========
x = customer_status[features].fillna(0)
churn_probabilities = model.predict_proba(x)[:,1]

# adding to the dataframe
customer_status['probabilities'] = churn_probabilities

#pd.cut will divide the data into particular segments

customer_status['risk segment'] = pd.cut(
    customer_status['probabilities'],
    bins = [0,0.3,0.7,1.0],
    labels=['🟢 Stable', '🟡 At Risk', '🔴 Critical']
)

print(customer_status)

# ✅ ========== ANALYSIS & OUTPUT ==========

# visualization of the outputs

total = len(customer_status['risk segment'])
risk_counts = (customer_status['risk segment'].value_counts()/total)*100

plt.figure()
plt.bar(risk_counts.index, risk_counts.values)
plt.xlabel("Risk Segment")
plt.ylabel("Number of Customers")
plt.title("Customer Churn Risk Distribution")
plt.show()

# High risk customer
high_risk = customer_status[customer_status['probabilities']>0.7].copy()
print("\n" + "-"*60)
print(f"HIGH RISK CUSTOMERS (>70% churn probability)")
print("-"*60)
print(f"Total Number of High Risk Customers: {len(high_risk)}")
print(f"Percentage: {(len(high_risk)/len(customer_status))*100:.1f}%")
print(f"Revenue at Risk: ${(high_risk['total_revenue']).sum():.2f}")

# moderate risk customer
medium_risk = customer_status[(customer_status['probabilities']>0.3) & (customer_status['probabilities']<=0.7)].copy()
print("\n" + "-"*60)
print(f"MEDIUM RISK CUSTOMERS (30% TO 70% churn probability)")
print("-"*60)
print(f"Total Number of Medium Risk Customers: {len(medium_risk)}")
print(f"Percentage: {(len(medium_risk)/len(customer_status))*100:.1f}%")
print(f"Revenue at Risk: ${(medium_risk['total_revenue']).sum():.2f}")

print('-'*60)

# revenue estimates of retaining atleast 30% of the high risk customers
total_spending = len(high_risk)*100
print("Total Spending")
print(total_spending)
estimated_revenue_saved = high_risk['total_revenue'].sum()*0.3
print("Estimated save")
print(estimated_revenue_saved)
print('='*60)
print("Estimated Revnue Saved")
print(estimated_revenue_saved-total_spending)
print('='*60)
