import sys
import os
import base64
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import pandas as pd
import plotly.express as px
from data_processing_pipeline import run_clean_pipeline
from visualizer import render_kpi_section, render_sales_revenue_section, Customer_Analytics_Kpi, render_risk_analysis, render_risk_analysis_2, render_risk_analysis_3, clean_data_for_ml_prediction, render_churn_prediction_model, visualize_churn_data, render_sales_revenue_section_2, render_sales_revenue_section_3, render_risk_analysis_4, render_risk_analysis_5, render_risk_analysis_6, call_churn, data_prep, executive_kpi_section
from utils import detect_anomalies
from commodity.data_fetching import essential_commodities_Clothing_Textiles, essential_commodities_electronics, essential_commodities_Food_and_Grocery, get_data
import plotly.graph_objects as go
from visualizer import prepare_data_for_prophet, call_prophet

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="E-Commerce Analytics Pro",
    page_icon=r"C:\Users\Lenovo\OneDrive\图片\Screenshots\Screenshot 2026-02-11 225832.png",

)

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def getbase64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


st.markdown("""
    <style>
    .custom-card {
        background-color: #26303C;
        border: 1px solid #3E4C59;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.title("Navigation")
    st.info("E-Commerce Analytics Pro v1.0")
    app_mode = st.sidebar.selectbox(
        "Select Analysis Mode",
        ["Standard Dashboard", "Strategic Insights", "Predictive Insights", "Commodity Tracker"]
    )

    uploaded_file = st.file_uploader("Upload your csv file", type=['csv'])

    # ── KEY FIX: store df in session state on upload ──
    if uploaded_file is not None:
        if 'filename' not in st.session_state or st.session_state.filename != uploaded_file.name:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state.filename = uploaded_file.name
            # clear old forecast when new file uploaded
            for key in ['forecast', 'forecast_days', 'prophet_df']:
                if key in st.session_state:
                    del st.session_state[key]
        st.success("File uploaded Successfully")

    # forecast slider only shown on Predictive Insights
    if app_mode == "Predictive Insights":
        st.header("Settings")
        forecast_days = st.slider("Forecast days beyond data", 7, 90, 30, step=7)
    else:
        forecast_days = 30

# ─────────────────────────────────────────
# COMMODITY TRACKER
# ─────────────────────────────────────────
if app_mode == "Commodity Tracker":
    
    st.title("Essential Commodities Tracker")
    st.sidebar.header("Commodity Filters")

    commodity_categories = {
        "Electronics": essential_commodities_electronics,
        "Food & Grocery": essential_commodities_Food_and_Grocery,
        "Clothing & Textiles": essential_commodities_Clothing_Textiles
    }

    select_category = st.selectbox("Select Category", options=list(commodity_categories.keys()))
    active_commodities = commodity_categories[select_category]

    selected_commodities = st.sidebar.multiselect(
        "Select Commodities",
        options=list(active_commodities.keys()),
        default=list(active_commodities.keys())[:2]
    )
    period   = st.sidebar.selectbox("Select Period",   ["1mo", "3mo", "6mo", "1y"], index=2)
    interval = st.sidebar.selectbox("Select Interval", ["1d", "1wk"], index=0)

    data_frames = []
    for name in selected_commodities:
        try:
            df_commodity = get_data(active_commodities[name], name, period, interval)
            data_frames.append(df_commodity)
        except Exception as e:
            st.warning(f"Failed to fetch data for {name}: {e}")

    if not data_frames:
        st.error("No data to display. Check selections or internet connection.")
        st.stop()

    data = pd.concat(data_frames, axis=1).ffill()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    st.subheader("Commodity Prices Over Time")
    fig = px.line(data, x=data.index, y=data.columns, title="Commodity Price Over Time")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Normalized Prices Comparison (Base=100)")
    normalized = data / data.iloc[0] * 100
    fig_norm = px.line(normalized, x=normalized.index, y=normalized.columns, title="Normalized Prices")
    st.plotly_chart(fig_norm, use_container_width=True)

# ─────────────────────────────────────────
# MAIN APP — requires uploaded data
# ─────────────────────────────────────────
elif 'df' not in st.session_state:
    st.header("Welcome!")
    st.write("Upload your CSV file from the sidebar to begin the analysis.")

else:
    df = st.session_state.df

    # ── STANDARD DASHBOARD ──────────────────────────
    if app_mode == "Standard Dashboard":
        st.title("E-Commerce Analytics Pro")
        cleaned_df    = run_clean_pipeline(df)
        anomaly_count = detect_anomalies(df)

        st.caption("Revenue")
        render_kpi_section(cleaned_df)
        st.caption("Sales Data")
        render_sales_revenue_section(cleaned_df)
        render_sales_revenue_section_2(cleaned_df)
        render_sales_revenue_section_3(cleaned_df)
        Customer_Analytics_Kpi(cleaned_df)

        if anomaly_count > 0:
            st.sidebar.warning(f"Note: {anomaly_count} anomalies were filtered.")
        st.success("Data Engine Loaded Successfully!")

    # ── STRATEGIC INSIGHTS ──────────────────────────
    elif app_mode == "Strategic Insights":
        st.title("Risk Analysis")
        cleaned_df = run_clean_pipeline(df)
        st.caption("Risk analysis page")
        render_risk_analysis(df)
        render_risk_analysis_2(df)
        render_risk_analysis_3(df)
        render_risk_analysis_4(df)
        render_risk_analysis_5(df)
        render_risk_analysis_6(df)

    # ── PREDICTIVE INSIGHTS ─────────────────────────
    elif app_mode == "Predictive Insights":
        st.title("Churn Analyzer")
        st.caption("Predictive analysis page")

        churned_data = call_churn(df)
        counts = churned_data['risk_segment'].value_counts().reset_index()
        counts.columns = ['risk_segment', 'count']

        fig = px.pie(
            counts,
            names='risk_segment',
            values='count',
            hole=0.55,
            title='Customer Risk Distribution',
            color='risk_segment',
            color_discrete_map={
                '🟢 Stable'  : "#00FF6A",
                '🟡 At Risk' : '#F4D03F',
                '🔴 Critical': "#FF1F06"
            }
        )
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            pull=[0, 0.05, 0.1]
        )
        fig.update_layout(showlegend=True, title_font_size=24, title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)

        def highlight_probability(val):
            if val >= 0.7:
                return 'background-color: #ff4d4d; color: white;'
            elif val >= 0.3:
                return 'background-color: #ffd633; color: black;'
            else:
                return 'background-color: #66cc66; color: white;'

        high_risk_customers     = churned_data[churned_data['risk_segment'] == '🔴 Critical']
        moderate_risk_customers = churned_data[churned_data['risk_segment'] == '🟡 At Risk']

        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        if not high_risk_customers.empty:
            styled_high = high_risk_customers.style.applymap(
                highlight_probability, subset=['churn_probability']
            )
            st.dataframe(styled_high, use_container_width=True)
        else:
            st.warning("No high risk customers found")
            
        st.subheader("🔴 High Risk Customers")
        st.dataframe(styled_high, use_container_width=True)
        st.download_button(
            label="Download High Risk Customers CSV",
            data=convert_df(high_risk_customers),
            file_name='high_risk_customers.csv',
            mime='text/csv'
        )

        if not moderate_risk_customers.empty:
            styled_moderate = moderate_risk_customers.style.applymap(
                highlight_probability, subset=['churn_probability']
            )
            st.dataframe(styled_moderate, use_container_width=True)
        else:
            st.warning("No moderate risk customers found")
        st.subheader("🟡 Moderate Risk Customers")  
        st.dataframe(styled_moderate, use_container_width=True)
        st.download_button(
            label="Download Moderate Risk Customers CSV",
            data=convert_df(moderate_risk_customers),
            file_name='moderate_risk_customers.csv',
            mime='text/csv'
        )

        executive_kpi_section(churned_data)

        # ── Revenue Forecast ────────────────────────
        st.title("Revenue Forecast")

        required = {'order-date', 'total-value', 'status'}
        missing  = required - set(df.columns)
        if missing:
            st.error(f"Missing columns in your CSV: {missing}")
            st.stop()

        with st.expander("Preview raw data"):
            st.dataframe(df.head(20), use_container_width=True)

        # Run forecast only if not cached or forecast_days changed
        if ('forecast' not in st.session_state or
                st.session_state.get('forecast_days') != forecast_days):
            with st.spinner("Running Prophet forecast..."):
                try:
                    st.session_state.forecast      = call_prophet(df, forecast_days=forecast_days)
                    st.session_state.forecast_days = forecast_days
                    st.session_state.prophet_df    = prepare_data_for_prophet(df)
                except Exception as e:
                    st.error(f"Forecast failed: {e}")
                    st.stop()

        forecast   = st.session_state.forecast
        prophet_df = st.session_state.prophet_df

        st.success("Forecast generated!")
        st.subheader("Revenue Forecast")

        last_date = prophet_df['ds'].max()

        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
            y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(211,138,46,0.15)',
            line=dict(color='rgba(0,0,0,0)'),
            name='95% confidence band'
        ))
        fig2.add_trace(go.Scatter(
            x=prophet_df['ds'], y=prophet_df['y'],
            name='Actual revenue',
            line=dict(color='#378ADD', width=1.5)
        ))
        fig2.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat'],
            name='Forecast',
            line=dict(color='#D85A30', width=2)
        ))
        fig2.add_trace(go.Scatter(
            x=[str(last_date), str(last_date)],
            y=[prophet_df['y'].min(), prophet_df['y'].max()],
            mode='lines',
            line=dict(color='#9ca3af', width=1.5, dash='dash'),
            name='forecast starts'
        ))
        fig2.update_layout(
            xaxis_title='Date',
            yaxis_title='Revenue ($)',
            legend=dict(orientation='h', y=-0.2),
            template='plotly_white',
            height=450
        )
        st.plotly_chart(fig2, use_container_width=True)

        with st.expander("View forecast table"):
            future_only = forecast[forecast['ds'] > last_date][
                ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
            ].copy()
            future_only.columns = ['Date', 'Forecast', 'Lower bound', 'Upper bound']
            future_only['Date'] = future_only['Date'].dt.date
            for col in ['Forecast', 'Lower bound', 'Upper bound']:
                future_only[col] = future_only[col].map('${:,.0f}'.format)
            st.dataframe(future_only, use_container_width=True)
