import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from visualizer import prepare_data_for_prophet, call_prophet

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Revenue Forecaster",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Revenue Forecast — Prophet")
st.markdown("Upload your ecommerce CSV to generate a revenue forecast.")

# ─────────────────────────────────────────
# SIDEBAR — FILE UPLOAD
# ─────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    forecast_days = st.slider("Forecast days beyond data", 7, 90, 30, step=7)

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if uploaded_file is None:
    st.info("Upload your CSV file from the sidebar to get started.")
    st.markdown("**Required columns:** `order-date`, `total-value`, `status`")
    st.stop()

# Load data
df = pd.read_csv(uploaded_file)

# Validate columns
required = {'order-date', 'total-value', 'status'}
missing  = required - set(df.columns)
if missing:
    st.error(f"Missing columns in your CSV: {missing}")
    st.stop()

st.success(f"✅ Loaded {len(df):,} rows")

# Show raw data preview
with st.expander("Preview raw data"):
    st.dataframe(df.head(20), use_container_width=True)

# ─────────────────────────────────────────
# RUN PROPHET
# ─────────────────────────────────────────
with st.spinner("Running Prophet forecast..."):
    try:
        forecast = call_prophet(df, forecast_days=forecast_days)
    except Exception as e:
        st.error(f"Forecast failed: {e}")
        st.stop()

st.success("✅ Forecast generated!")

# ─────────────────────────────────────────
# FORECAST CHART
# ─────────────────────────────────────────
st.subheader("Revenue Forecast")

fig = go.Figure()

# Historical data (aggregate for display)
prophet_df = prepare_data_for_prophet(df)

fig.add_trace(go.Scatter(
    x=prophet_df['ds'],
    y=prophet_df['y'],
    name='Actual revenue',
    line=dict(color='#378ADD', width=1.5)
))

# Confidence band
fig.add_trace(go.Scatter(
    x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
    y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
    fill='toself',
    fillcolor='rgba(211,138,46,0.15)',
    line=dict(color='rgba(0,0,0,0)'),
    name='95% confidence band'
))

# Forecast line
fig.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat'],
    name='Forecast',
    line=dict(color='#D85A30', width=2)
))

# Vertical line at last known date
last_date = prophet_df['ds'].max()
fig.add_trace(go.Scatter(
    x=[str(last_date), str(last_date)],
    y=[prophet_df['y'].min(), prophet_df['y'].max()],
    mode='lines',
    line=dict(color='#9ca3af', width=1.5, dash='dash'),
    name='forecast starts'
))
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Revenue ($)',
    legend=dict(orientation='h', y=-0.2),
    template='plotly_white',
    height=450
)

st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────
# FORECAST TABLE
# ─────────────────────────────────────────
with st.expander("View forecast table"):
    future_only = forecast[forecast['ds'] > last_date][
        ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
    ].copy()
    future_only.columns = ['Date', 'Forecast', 'Lower bound', 'Upper bound']
    future_only['Date'] = future_only['Date'].dt.date
    for col in ['Forecast', 'Lower bound', 'Upper bound']:
        future_only[col] = future_only[col].map('${:,.0f}'.format)
    st.dataframe(future_only, use_container_width=True)