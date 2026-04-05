import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

data = pd.read_csv('Sample4_data.csv')
temp = data.copy() ## to keep original data safe

# eda
print(data.describe())
print(data.info())
print(data.isnull().sum())

# feature engineering

# removing the rows containing returned and cancelled
temp = temp[~temp['status'].str.contains('Returned|Cancelled',case=False,na=False)] ## important line
print(temp)

daily = (
    temp.groupby('order-date')
    .agg(
        revenue = ('total-value','sum'),
        num_orders = ('order-id','count'),
        avg_order_val = ('total-value','mean')
    )
    .reset_index()
    .rename(columns={'order-date':'date'})
    .sort_values('date',ascending=True)
    .reset_index(drop=True)
)

print(daily)
print(daily.columns.to_list())

#   Finding missing dates with 0 

# full range of dates
daily["date"] = pd.to_datetime(daily["date"])
full_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
daily = daily.set_index("date").reindex(full_range, fill_value=0).reset_index()
daily.rename(columns={"index": "date"}, inplace=True)
 
print(f"Daily revenue shape: {daily.shape}")
print(daily[["date", "revenue", "num_orders"]].head(5).to_string(index=False))

# calender features

# days of the week
daily['week_day'] = daily['date'].dt.dayofweek

# weekend 
daily['is_weekend'] = (daily['week_day']>=5).astype(int)

# month (but in numbers)
daily['month'] = daily['date'].dt.month 

# quarter (1-4)
daily['quarter'] = daily['date'].dt.quarter

# day of the month
daily['day_of_the_month'] = daily['date'].dt.day

# day of the year
daily['day_of_the_year'] = daily['date'].dt.dayofyear

# Week of year (1–52)
daily["week_of_year"]   = daily["date"].dt.isocalendar().week.astype(int)

# is it month start or month end
daily['is_month_start'] = daily['date'].dt.is_month_start.astype(int)
daily['is_month_end'] = daily['date'].dt.is_month_end.astype(int)

daily["is_festive"] = daily["month"].isin([10, 11, 12]).astype(int)

# lag features 
# this help model to understand what was the revenue past n days before (eg for today and n = 4 what was the revenue 4 days back)

for lag in [1,2,3,7,14,21,30]:
    daily[f'lag_{lag}d'] = daily['revenue'].shift(lag)

# now rolling functions (to get the average or median values of the periods eg 7 day period or 14 day period or 30 day period)
daily['rolling_mean_7d'] = daily['revenue'].shift(1).rolling(7,min_periods=1).mean()
daily['rolling_mean_14d'] = daily['revenue'].shift(1).rolling(14,min_periods=1).mean()
daily['rolling_mean_30d'] = daily['revenue'].shift(1).rolling(30,min_periods=1).mean()

print(daily['rolling_mean_7d'])

# now volatility features
daily['rolling_std_7d'] = daily['revenue'].shift(1).rolling(7,min_periods=1).std().fillna(0)
daily['rolling_std_14d'] = daily['revenue'].shift(1).rolling(14,min_periods=1).std().fillna(0)

# revenue momentum
daily['momentum_7d'] = daily['lag_1d']/(daily['rolling_mean_7d']+1e-9)


# Week-over-week change
daily["wow_change"] = daily["revenue"].shift(1) - daily["revenue"].shift(8)
 
# Month-over-month change (approx)
daily["mom_change"] = daily["revenue"].shift(1) - daily["revenue"].shift(31)

## category level features

# revenue breakdown by category
cat_daily = (
    data[~data['status'].isin(['Cancelled','Returned'])]
    .groupby(['order-date','product-category'])['total-value']
    .sum()
    .unstack(fill_value=0)
    .reset_index()
    .rename(columns={'order-date':'date'})
)

cat_daily.columns = (
    ["date"] + [f"rev_{c.lower()}" for c in cat_daily.columns[1:]]
)

# Reindex to full date range and merge
cat_daily = cat_daily.set_index("date").reindex(full_range, fill_value=0).reset_index()
cat_daily.rename(columns={"index": "date"}, inplace=True)
 
daily = daily.merge(cat_daily, on="date", how="left")
 
print("✅ Category revenue features added:", [c for c in daily.columns if c.startswith("rev_")])

before = len(daily)
daily.dropna(inplace=True)
daily.reset_index(drop=True, inplace=True)
print(f"\n✅ Dropped {before - len(daily)} rows with NaN from lag warmup")
 
 
daily.to_csv("featured_revenue.csv", index=False)
 
print(f"\n✅ Saved → featured_revenue.csv")
print(f"   Shape  : {daily.shape}")
print(f"   Columns: {list(daily.columns)}")
print(f"\n📋 Sample (first 3 rows):")
print(daily[["date","revenue","lag_7d","rolling_mean_7d",
             "is_weekend","month","momentum_7d"]].head(3).to_string(index=False))

