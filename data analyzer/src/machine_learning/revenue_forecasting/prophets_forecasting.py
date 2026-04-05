import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error,mean_squared_error
import plotly.graph_objects as go
import pickle
import os


# now we import the data

raw_data = pd.read_csv("Sample4_data.csv")
raw_data = raw_data[~raw_data['status'].isin(['Cancelled','Returned'])]
print(raw_data)

# now structuring the data according to the prophet model requirements
prophet_df = (
    raw_data.groupby('order-date')['total-value'].sum().reset_index().rename(columns={'order-date':'ds','total-value':'y'}).sort_values('ds').reset_index(drop=True)
)

prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
 
print(f"Data ready for Prophet: {prophet_df.shape}")
print(prophet_df.head(5).to_string(index=False))

# now splitting the date
split_date = prophet_df['ds'].quantile(0.8)

train_df = prophet_df[prophet_df['ds']<split_date]
test_df = prophet_df[prophet_df['ds']>=split_date]

print(f"\nTrain: {len(train_df)} rows ({train_df['ds'].min().date()} → {train_df['ds'].max().date()})")
print(f"   Test : {len(test_df)}  rows ({test_df['ds'].min().date()} → {test_df['ds'].max().date()})")

model = Prophet(
    yearly_seasonality       = True,
    weekly_seasonality       = True,
    daily_seasonality        = False,
    seasonality_mode         = 'multiplicative',
    changepoint_prior_scale  = 0.05,
    seasonality_prior_scale  = 10,
    interval_width           = 0.95
)

model.add_seasonality(
    name          = 'festive_season',
    period        = 365.25,
    fourier_order = 5
)

model.fit(train_df)
print("\nProphet model trained successfully")

future = model.make_future_dataframe(
    periods = len(test_df),
    freq    = 'D'
)
print(f"\nFuture dataframe: {len(future)} rows")
print(f"   Covers: {future['ds'].min().date()} → {future['ds'].max().date()}")

forecast = model.predict(future)
 
print(f"\n✅ Forecast generated — {len(forecast)} rows")
print("\n📋 Sample predictions (last 5 rows):")
print(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(5).to_string(index=False))


# ─────────────────────────────────────────
# STEP 8 — Extract test predictions
# ─────────────────────────────────────────
test_forecast = (
    forecast[forecast['ds'] >= split_date]
    [['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    .merge(test_df, on='ds', how='inner')
)
train_forecast = (
    forecast[forecast['ds'] < split_date]
    [['ds', 'yhat']]
    .merge(train_df, on='ds', how='inner')
)
 
def evaluate(y_true, y_pred, label):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    r2   = 1 - (np.sum((y_true - y_pred)**2) /
                np.sum((y_true - y_true.mean())**2))
    print(f"\n📊 {label}")
    print(f"   MAE  : ${mae:,.2f}")
    print(f"   RMSE : ${rmse:,.2f}")
    print(f"   MAPE : {mape:.2f}%  ← lower is better")
    print(f"   R²   : {r2:.4f}    ← closer to 1 is better")
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}
 
train_metrics = evaluate(train_forecast['y'], train_forecast['yhat'], "Train set")
test_metrics  = evaluate(test_forecast['y'],  test_forecast['yhat'],  "Test set")
 

fig = go.Figure()
 
fig.add_trace(go.Scatter(
    x    = pd.concat([test_forecast['ds'], test_forecast['ds'][::-1]]),
    y    = pd.concat([test_forecast['yhat_upper'], test_forecast['yhat_lower'][::-1]]),
    fill = 'toself',
    fillcolor = 'rgba(211,138,46,0.15)',
    line = dict(color='rgba(0,0,0,0)'),
    name = '95% confidence band'
))
fig.add_trace(go.Scatter(
    x=train_df['ds'], y=train_df['y'],
    name='Actual (train)', line=dict(color='#378ADD', width=1.5)
))
fig.add_trace(go.Scatter(
    x=train_forecast['ds'], y=train_forecast['yhat'],
    name='Prophet fit (train)', line=dict(color='#378ADD', width=1, dash='dot')
))
fig.add_trace(go.Scatter(
    x=test_forecast['ds'], y=test_forecast['y'],
    name='Actual (test)', line=dict(color='#1D9E75', width=2)
))
fig.add_trace(go.Scatter(
    x=test_forecast['ds'], y=test_forecast['yhat'],
    name='Prophet forecast (test)', line=dict(color='#D85A30', width=2)
))
fig.add_vrect(
    x0=split_date, x1=prophet_df['ds'].max(),
    fillcolor='#1D9E75', opacity=0.05,
    annotation_text='test region', annotation_position='top left'
)
fig.update_layout(
    title='Prophet — actual vs forecast',
    xaxis_title='Date', yaxis_title='Revenue ($)',
    legend=dict(orientation='h', y=-0.2),
    template='plotly_white', height=500
)
fig.write_html("prophet_forecast.html")
print("\n✅ Forecast chart → prophet_forecast.html")
 
# ─────────────────────────────────────────
# STEP 11 — Plot components
#
# Prophet's most powerful feature —
# breaks forecast into separate components:
#   Trend        → overall revenue direction
#   Weekly       → Mon-Sun daily pattern
#   Yearly       → Jan-Dec seasonal pattern
#   Festive      → your custom seasonality
# ─────────────────────────────────────────
comp_fig = model.plot_components(forecast)
comp_fig.savefig("prophet_components.png", dpi=150, bbox_inches='tight')
print("✅ Components chart → prophet_components.png")
 
# ─────────────────────────────────────────
# STEP 12 — Bonus: forecast 30 days BEYOND
#           your existing data
# ─────────────────────────────────────────
future_30 = model.make_future_dataframe(periods=len(test_df) + 30, freq='D')
forecast_30 = model.predict(future_30)
future_only = forecast_30.tail(30)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
 
print(f"\n🔮 30-day future forecast:")
print(future_only[['ds','yhat']].to_string(index=False))
 
os.makedirs("models", exist_ok=True)
with open("prophet_revenue.pkl", "wb") as f:
    pickle.dump(model, f)
print("\n✅ Model saved → prophet_revenue.pkl")