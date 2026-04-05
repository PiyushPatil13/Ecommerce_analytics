import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from mitosheet.streamlit.v1 import spreadsheet
import pandas as pd
from datetime import datetime
from utils import detect_anomalies
from pandas import DatetimeIndex
import pickle
import os
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet


st.markdown("""
<style>
[data-testid="stVerticalBlock"] > [data-testid="stPlotlyChart"] {
    border-radius: 12px;
}
.card {
    background-color: #111827;
    border-radius: 12px;
    padding: 16px;
    height: 320px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)



with st.sidebar:
    st.title("")

def render_kpi_section(df):
    # 1. Calculate Metrics
    current_rev = df['total-value'].sum()
    avg_order = df['total-value'].mean()
    total_qty = df['quantity'].sum()
    anomaly_count = detect_anomalies(df)
    
    # 2. Layout: Create 3 columns
    col1, col2, col3,col4 = st.columns(4)    
    
    # 3. Design the Cards (Matching the UI Image)
    with col1:
        st.metric(
            label="Total Revenue", 
            value=f"${current_rev:,.0f}", 
            delta="+4.1%" ,# You can calculate this vs. last month's data
        )
        
    with col2:
        st.metric(
            label="Average Order Value", 
            value=f"${avg_order:,.2f}", 
            delta="-0.5%" 
        )
        
    with col3:
        st.metric(
            label="Items Sold", 
            value=f"{total_qty:,}", 
            delta="+12%"
        )
    with col4:
        st.metric(
            label = "Anomalies",
            value = f"{anomaly_count:,}", 
        )

def render_sales_revenue_section(df):

    col1, col2, col3 = st.columns(3)
    # now rendering pie chart
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Revenue By Product Category",divider="gray")
        categorical_data = df.groupby('product-category')['total-value'].sum().reset_index()

        # plotting 
        figure = px.pie(
            categorical_data,
            values='total-value',
            names='product-category',
            hole = 0.3,
            color_discrete_sequence=px.colors.sequential.Tealgrn_r,
            template="plotly_dark",
            height=220
        )

        figure.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=True
        )
        st.plotly_chart(figure,use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

       
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Daily Sales Trend",divider="gray")
        df['order-date'] = pd.to_datetime(df['order-date'])

    
        df = df.set_index('order-date')
        weekly_data = df['total-value'].resample('W').sum().reset_index()
        figure1 = px.area(weekly_data,x="order-date",y="total-value", height=220)
        figure1.update_traces(
            line_color = '#00D4FF',
            line_width = 3,
            fillcolor = 'rgba(0,212,255,0.15)',
            line_shape = 'spline',
        )
        figure1.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=True,
            template = "plotly_dark",
            height = 220,
            xaxis = dict(showgrid = False),
            yaxis = dict(showgrid = True,gridcolor = "#26303C",title = "Revenue ($)"),
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(figure1,use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Top 5 Performing Cities",divider="gray")
        top_performing_cities = df.groupby("city")['total-value'].sum().reset_index()
        top_performing_cities = (top_performing_cities.sort_values(by='total-value',ascending=False).head(5))
        figure2 = px.bar(top_performing_cities,x="city",y='total-value',color_discrete_sequence=px.colors.qualitative.Set2)
        figure2.update_traces(
        marker_color="#6EC7A8",
        marker_line_width=0
)
        figure2.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=True,
            height = 220,
            plot_bgcolor="#020617",
            xaxis=dict(showgrid=False),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.05)"
            ),
           
            
        )
        st.plotly_chart(figure2,use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


def render_sales_revenue_section_2(df):
    col1,spacer,col2,spacer,col3 = st.columns([1,0.10,1,0.10,1])
    with col1:
    # month over month growth
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Monthly Revenue Heatmap",divider="gray")
        # change if month number to month name
        df['month'] = df['order-date'].dt.month
        df['month-name'] = df['order-date'].dt.strftime('%b')
        monthly = df.groupby(['month','month-name'])['total-value'].sum().reset_index().sort_values('month')
        monthly['Mon-growth'] = monthly['total-value'].pct_change()*100
        figure = px.bar(monthly.dropna(),color ='Mon-growth',x='month-name',y='Mon-growth',title='Monthly Growth Trend',color_continuous_scale=['red','yellow','green'],text=monthly.dropna()['Mon-growth'].apply(
        lambda x: f"+{x:.1f}%" if x > 0 else f"{x:.1f}%"
    ))
        
        st.plotly_chart(figure,use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # order status breakdown
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Order status",divider="gray")
        status_counts = df['status'].value_counts()
        figure = px.pie(status_counts,values = status_counts.values,names=status_counts.index,hole = 0.2,title='Order Status Distribution',color_discrete_sequence=px.colors.sequential.Tealgrn_r,template="plotly_dark",height=320)
        figure.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=True
        )
        st.plotly_chart(figure,use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)






def render_sales_revenue_section_3(df):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Revenue Heatmap",divider="gray")
    df['day'] = pd.to_datetime(df['order-date']).dt.day_name()
    df['month'] = pd.to_datetime(df['order-date']).dt.month_name()
    heatmap = df.groupby(['month','day'])['total-value'].sum().unstack()
    month_order = ['January', 'February', 'March', 'April', 
                    'May', 'June', 'July', 'August',
                    'September', 'October', 'November', 'December']
    heatmap = heatmap.reindex(month_order)
    fig = px.imshow(heatmap,title="Revenue Heatmap by Day and Month",color_continuous_scale='YlOrRd',aspect='auto')
    fig.update_layout(
        height = 500,
        width = 1000,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(20,28,46,0.8)',
        font=dict(color='#8B9AC0'),
    
        title=dict(
            font=dict(
                family="Syne, sans-serif",
                size=18,
                color="#F0F4FF"
            )
        )
    )
    

    st.plotly_chart(fig,use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

        
        




def Customer_Analytics_Kpi(df):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Unique Customer",divider="gray")
        unique_customer_data = df['customer-name'].nunique()
        st.metric(
            label= "Unique Customer Count",
            value=f"{unique_customer_data:,.0f}"
        )

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Loyal Customer",divider="gray")
        total_customers = df['customer-email'].nunique()
        customer_orders = (
            df.groupby('customer-email')['order-id'].nunique().reset_index(name = 'order_count')
        )
        repeat_customers = customer_orders[customer_orders['order_count']>1]
        repeat_customers_count = repeat_customers.shape[0]
        repeat_customer_rate = (repeat_customers_count/total_customers)*100
        st.metric(
            label= "Loyal Customer Count",
            value=f"{repeat_customer_rate:,.0f} %"
        )   

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Customer Value",divider="gray")
        customer_value = (
            df.groupby('customer-email')['total-value'].sum().reset_index(name='customer_lifetime_value')
        )
        average_customer_value = customer_value['customer_lifetime_value'].mean()
        st.metric(
            label="Customer Lifetime Value",
            value=f"$ {average_customer_value:,.0f}"
        )

def render_risk_analysis(df):

    col1,spacer,col2 = st.columns([1,0.12,1])
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Product Diversification",divider="gray")
        total_revenue = df['total-value'].sum()
        category_revenue = df.groupby('product-category')['total-value'].sum().reset_index()
            
        category_revenue['pct'] = (category_revenue['total-value']/total_revenue)*100
        top_category = category_revenue.sort_values(
            by = 'pct',ascending = False,
        ).iloc[0]

        severity_color = (
            "red" if top_category['pct'] >= 50
            else "orange" if top_category['pct'] >= 33
            else "green"
        )

        st.markdown(
            f"""
            <div style="border-left: 6px solid {severity_color};
                        padding-left: 12px;">
                <b>{top_category['product-category']}</b> contributes
                <b>{top_category['pct']:.1f}%</b> of total revenue
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(min(top_category['pct'] / 100, 1.0))

        if top_category['pct'] >= 50:
            st.caption("📌 Recommendation: Diversify revenue by scaling mid-tier categories.")
        elif top_category['pct'] >= 33:
            st.caption("📌 Recommendation: Monitor category concentration monthly.")
        else:
            st.caption("✅ No action required.")

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Cities Diversification",divider="gray")
        total_revenue = df['total-value'].sum()
        Geographic_Revenue = df.groupby('city')['total-value'].sum().reset_index()
        top_5_revenue_df = Geographic_Revenue.sort_values(
                by = 'total-value',ascending = False
            ).iloc[:5]
        top_5_revenue = top_5_revenue_df['total-value'].sum()
        percentage = (top_5_revenue/total_revenue)*100
        
        if percentage >= 35:
            st.error(
                f"📍 **High Geographic Concentration**\n\n"
                f"Top 5 cities contribute **{percentage:.1f}%** of revenue."
            )
            st.caption("📌 Recommendation: Expand marketing to Tier-2 / Tier-3 cities.")
        else:
            st.success(
                f"🌍 **Healthy Geographic Distribution**\n\n"
                f"Top 5 cities contribute only **{percentage:.1f}%** of revenue."
            )

        st.markdown('</div>', unsafe_allow_html=True)

def render_risk_analysis_2(df):
    col1,spacer,col2 = st.columns([1,0.12,1])
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Peak Month Analysis",divider="gray")
        df['order-date'] = pd.to_datetime(df['order-date'])
        df['month'] = df['order-date'].dt.to_period('M')

        monthly_revenue = (
            df.groupby('month')['total-value'].sum().reset_index()
        )

        # peak month calculation
        peak_month_revenue = monthly_revenue['total-value'].max()
        total_revenue = df['total-value'].sum()
        peak_month_revenue_pct = (peak_month_revenue/total_revenue)*100
        peak_month = int(monthly_revenue['total-value'].idxmax())
        days = {
            1:"January",
            2:"February",
            3:"March",
            4:"April",
            5:"May",
            6:"June",
            7:"July",
            8:"August",
            9:"September",
            10:"October",
            11:"November",
            12:"December"
        }

        if peak_month_revenue_pct >=25:
            st.write(f"Peak Month Revenue was achieved in {days[peak_month]} and it is {peak_month_revenue_pct:.0f}% which is 🔴 High")
        elif peak_month_revenue_pct >= 15 and peak_month_revenue_pct < 25:
            st.write(f"Peak Month Revenue was achieved in {days[peak_month]} and it is {peak_month_revenue_pct:.0f}% which is 🟡 Medium")
        else:
            st.write(f"Peak Month Revenue was achieved in {days[peak_month]} and it is {peak_month_revenue_pct:.0f}% which is 🟢 Low")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Seasonality Risk Analysis",divider="gray")
        top_3_month_revenue = (
            monthly_revenue.sort_values('total-value',ascending=False).head(3)['total-value'].sum()
        )
        total_revenue = df['total-value'].sum()
        seasonality_risk = (top_3_month_revenue/total_revenue)*100

        if seasonality_risk >= 55:
            st.write(f"🔴 High Seasonal Dependency — {seasonality_risk:.0f}%")
        elif seasonality_risk >=45 and seasonality_risk < 55:
            st.write(f"🟡 Medium Seasonal Dependency — {seasonality_risk:.0f}%")
        else:
            st.write(f"🟢 Low Seasonal Dependency — {seasonality_risk:.0f}%")
        st.markdown('</div>', unsafe_allow_html=True)


def render_risk_analysis_3(df):
    col1,spacer,col2 = st.columns([1,0.12,1])
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Customer Concentration Risk",divider="gray")
        total_value = df['total-value'].sum()

        # top 1 percent income
        top_1_percent_revenue = df.groupby('customer-email')['total-value'].sum().nlargest(int(len(df)*0.01)).sum()
        top_1_percent_revenue_percentage = (top_1_percent_revenue/total_value)*100

        if top_1_percent_revenue_percentage >= 20:
            st.write(f"⚠️ **High Concentration Risk:** The top 1% of customers contribute "
        f"{top_1_percent_revenue_percentage:.0f}%. The business is highly dependent "
        f"on a few key accounts." )
        else:
            st.write(f"✅ **Revenue Stability:** The top 1% contribute only "
        f"{top_1_percent_revenue_percentage:.0f}%. Our revenue base is "
        f"well-diversified with low dependency on high-tier outliers.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Margin Erosion & Volatility",divider="gray")

        mean_value = df['total-value'].mean()
        standard_deviation = df['total-value'].std()
        Coefficient_of_Variation = standard_deviation/mean_value

        if Coefficient_of_Variation >= 0.5:
            st.write(f"High CV {Coefficient_of_Variation:.2f}. Your revenue is spiky ⚠️. You have huge days followed by dead days. This is a Cash Flow Risk.")
        else:
            st.write(f"Low CV {Coefficient_of_Variation:.2f} .Your revenue is stable and predictable. This is a Healthy Business.")

        st.markdown('</div>', unsafe_allow_html=True)

def render_risk_analysis_4(df):
    # col1,space,col2 = st.columns([1,0.12,1])
    
    # return rate by category
    # with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    returns = df[df['status']=='Returned'].groupby('product-category').size()
    total = df.groupby('product-category').size()
    return_rate = ((returns/total)*100).round(2).sort_values(ascending = False)
    st.subheader("📦 Return Rate By Category",divider="gray")
    cols = st.columns(len(return_rate))
    return_df = return_rate.reset_index()
    return_df.columns = ['Category', 'Return Rate (%)']
    return_df['Risk Level'] = return_df['Return Rate (%)'].apply(
    lambda x: 'High Risk' if x >= 25 else 'Moderate Risk'
    )
    for col,(category,rate) in zip(cols,return_rate.items()):
        if rate>25:
            color = "🔴 High"
        elif rate >23:
            color = "🟠 Medium"
        else:
            color = "🟢 Low"
        
        col.metric(
            label=category,
            value=f"{rate}%",
            delta = color
        )

    figure = px.bar(
        return_df,
        x=return_rate.index,
        y = return_rate.values,
        labels={'x': 'Category', 'y': 'Return Rate (%)'},
        title="Return Rate Comparison",
        text=return_rate.values,
        color='Risk Level',
        color_discrete_map={
        'High Risk': '#EF553B',      # red
        'Moderate Risk': '#FFA15A'   # orange
        },
        width=200,
    )
    # now adding benchmark line to highlight the limit 
    figure.add_hline(
        y=25,
        line_dash="dash",
        line_color="white",
        annotation_position="top right",
        annotation_text="25% Risk Threshold",

    )
    figure.update_traces(texttemplate='%{text:.2f}%', textposition='inside', insidetextanchor='middle',textfont=dict(color= 'black',size = 16))
    figure.update_layout(template="plotly_dark")

    st.plotly_chart(figure, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    

def render_risk_analysis_5(df):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🤐 Volatility Analysis",divider="gray",text_alignment='center')
    # 1. Ensure date conversion and sorting (Critical for rolling windows)
    df['order-date'] = pd.to_datetime(df['order-date'], errors='coerce')
    df = df.dropna(subset=['order-date']).sort_values('order-date')
    
    # 2. Resample to weekly and calculate stats
    weekly = df.groupby(pd.Grouper(key='order-date', freq='W'))['total-value'].sum()
    # Filter for positive revenue and handle potential gaps
    weekly = weekly[weekly > 0]
    
    rolling_mean = weekly.rolling(window=4).mean()
    rolling_std = weekly.rolling(window=4).std()
    
    # Define Upper and Lower bounds
    upper_bound = rolling_mean + rolling_std
    lower_bound = rolling_mean - rolling_std

    fig = go.Figure()

    # 3. Upper Band (Note: x must be the index/dates)
    fig.add_trace(go.Scatter(
        x=weekly.index, 
        y=upper_bound,
        line=dict(width=0),
        marker=dict(size=0),
        showlegend=False,
        name='Upper Band'
    ))

    # 4. Lower Band with Fill (Shaded Area)
    fig.add_trace(go.Scatter(
        x=weekly.index,
        y=lower_bound,
        line=dict(width=0),
        marker=dict(size=0),
        fill='tonexty', # Fills the area between Upper and Lower
        fillcolor='rgba(0, 212, 255, 0.1)',
        showlegend=False,
        name='Volatility Range'
    ))

    # 5. The Main Average Line
    fig.add_trace(go.Scatter(
        x=weekly.index,
        y=rolling_mean,
        name='4-Week Moving Avg',
        line=dict(color='#00d4ff', width=3)
    ))

    # 6. Actual Weekly Revenue (Optional but recommended for context)
    fig.add_trace(go.Scatter(
        x=weekly.index,
        y=weekly,
        name='Weekly Revenue',
        mode='markers',
        marker=dict(color='#8B9AC0', size=4, opacity=0.5)
    ))

    fig.update_layout(
        title='Revenue Volatility Analysis (Bollinger Bands)',
        yaxis_title='Revenue ($)',
        xaxis_title='Date',
        template='plotly_dark', # Simplifies the styling
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(20,28,46,0.8)',
        hovermode='x unified',
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_risk_analysis_6(df):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🛒 Category Revenue Over Time",divider="gray",text_alignment='center')
    df['order-date'] = pd.to_datetime(df['order-date'], errors='coerce')
    df['month'] = df['order-date'].dt.strftime('%b') # jan ,feb ,march
    df['month-num'] = df['order-date'].dt.month # 1,2,3 for sorting
    category_monthly = df.groupby(['month-num','month','product-category'])['total-value'].sum().reset_index()
    # sorting the months
    category_monthly = category_monthly.sort_values('month-num')
    fig = go.Figure()
    for category in category_monthly['product-category'].unique():
        cat_data = category_monthly[
            category_monthly['product-category']==category
        ].copy()
        color_discrete_map = {
            'Electronics': '#00D4FF',
            'Grocery':     '#00FFB2',
            'Clothing':    '#7B61FF',
            'Decorative':  '#FFB347'
        }

        # Apply per trace
        colors = {
            'Electronics': '#00D4FF',
            'Grocery':     '#00FFB2',
            'Clothing':    '#7B61FF',
            'Decorative':  '#FFB347'
        }

        fig.add_trace(go.Scatter(
            x=cat_data['month'],
            y=cat_data['total-value'],
            name=category,
            mode='lines+markers',
            line=dict(width=2)
        ))
        mean_revenue = cat_data['total-value'].mean()
        std_dev_revenue = cat_data['total-value'].std()
        for _,row in cat_data.iterrows():
            deviation = row['total-value'] - mean_revenue
            if deviation > 1.5 * std_dev_revenue:
                fig.add_annotation(
                    x=row['month'],
                    y=row['total-value'],
                    text=f"▲ {category} peak",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor='#00FFB2',
                    font=dict(
                        color='#00FFB2',
                        size=10
                    ),
                    bgcolor='rgba(0,255,178,0.1)',
                    bordercolor='#00FFB2',
                    borderwidth=1,
                    ay=-40  # arrow points up
                )
            elif deviation < -1.5 * std_dev_revenue:
                fig.add_annotation(
                    x=row['month'],
                    y=row['total-value'],
                    text=f"▼ {category} dip",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor='#FF4D6D',
                    font=dict(
                        color='#FF4D6D',
                        size=10
                    ),
                    bgcolor='rgba(255,77,109,0.1)',
                    bordercolor='#FF4D6D',
                    borderwidth=1,
                    ay=40  # arrow points down
                )
    fig.update_layout(
        title='Category Revenue Over Time',
        yaxis_title='Revenue ($)',
        xaxis_title='Month',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(20,28,46,0.8)',
        font=dict(color='#8B9AC0'),
        xaxis=dict(gridcolor='rgba(0,212,255,0.08)'),
        yaxis=dict(gridcolor='rgba(0,212,255,0.08)'),
        legend=dict(
            bgcolor='rgba(20,28,46,0.8)',
            bordercolor='rgba(0,212,255,0.15)',
            borderwidth=1
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)





        
def clean_data_for_ml_prediction(df):
    df['order-date'] = pd.to_datetime(df['order-date'])
    ref_date = df['order-date'].max()
    churned_df = df.groupby('customer-email').agg({
        'order-date': lambda x: (ref_date - x.max()).days,
        'order-id':'nunique',
        'total-value':'sum'
    }).reset_index()


    churned_df.columns = ['customer-email', 'Recency', 'Frequency', 'Total_Value']

    churned_df['Churn_target'] = (churned_df['Frequency']==1).astype('int')
    churned_df['Churn-Label'] = churned_df['Churn_target'].map({1:'Yes',0:'No'})
    return churned_df
    # print("Class Counts:")
    # print(churned_df['Churn_target'].value_counts())
    # print(churned_df['Frequency'].describe())
    column = ['Recency', 'Total_Value']
    X = churned_df[column]
    Y = churned_df['Churn_target']
    

def render_churn_prediction_model(df):
    # 1. Clean and aggregate the data first
    data = clean_data_for_ml_prediction(df)
    
    # 2. Define the path to your pre-trained model
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, "machine_learning", "churn_predictor1.pkl")
    
    # 3. Load the .pkl file
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # 4. Predict using the features (Recency and Total_Value)
    X = data[['Recency', 'Total_Value']]
    data['Churn_Prob'] = model.predict_proba(X)[:, 1]  # Probability of churn
    data['Predicted_Label'] = model.predict(X)         # 0 or 1
    
    return data

def visualize_churn_data(df):
    # Ensure we get the data from your model located in src/machine_learning/
    datas = render_churn_prediction_model(df)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Churn Analyzer", divider="gray")

    # Step 1: Create categories with explicit string default to avoid DType error
    # Note: Using 0.75 assumes your model outputs decimals
    conditions = [
        datas['Churn_Prob'] >= 0.75,
        (datas['Churn_Prob'] >= 0.60) & (datas['Churn_Prob'] < 0.75),
        (datas['Churn_Prob'] >= 0.25) & (datas['Churn_Prob'] < 0.60),
        datas['Churn_Prob'] < 0.25
    ]
    
    choices = ['Critical', 'High Risk', 'Medium', 'Stable']
    
    # Adding default='Stable' prevents the Integer vs String conflict
    datas['Category'] = np.select(conditions, choices, default='Stable')

    # Step 2: Ensure all categories are counted correctly
    # reset_index() naming varies; this approach is the most stable
    category_counts = datas['Category'].value_counts().reset_index()
    category_counts.columns = ['Category', 'Count']

    # Step 3: Plot with specific color mapping for the "Predictive Insights" page
    figure = px.pie(
        category_counts,
        values='Count',
        names='Category',
        hole=0.4,
        color='Category',
        color_discrete_map={
            'Critical': '#ff4b4b',   # Red
            'High Risk': '#ffa500',  # Orange
            'Medium': '#f0f2f6',     # Light Gray
            'Stable': '#28a745'      # Green
        },
        template="plotly_dark",
        height=300,
        
    )

    # Optional: Update layout for better visibility
    figure.update_layout(
        showlegend=True,
        margin=dict(t=0, b=0, l=0, r=0),
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)'
    )
    figure.update_traces(
        textinfo="percent+label",
        pull=[0, 0.05, 0.1]
    )

    st.plotly_chart(figure, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def data_prep(df):
    df = df.copy()

    # 🔥 normalize columns properly
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    # 🔥 convert date properly
    df['order-date'] = pd.to_datetime(df['order-date'])

    customer_status = df.groupby('customer-email').agg(
        last_purchase=('order-date', 'max'),
        total_orders=('order-id', 'count'),
        total_revenue=('total-value', 'sum'),
        average_order_value=('total-value', 'mean'),
        return_count=('status', lambda x: (x == 'Returned').sum()),
        cancel_count=('status', lambda x: (x == 'Cancelled').sum())
    ).reset_index()

    max_date = df['order-date'].max()

    customer_status['recency'] = (
        max_date - customer_status['last_purchase']
    ).dt.days

    customer_status['return_rate'] = (
        customer_status['return_count'] /
        customer_status['total_orders'].clip(lower=1)
    )

    customer_status['cancelled_rate'] = (
        customer_status['cancel_count'] /
        customer_status['total_orders'].clip(lower=1)
    )

    customer_first = df.groupby('customer-email')['order-date'].min().reset_index()
    customer_first.columns = ['customer-email', 'first-purchase']

    customer_status = customer_status.merge(
        customer_first, on='customer-email'
    )

    customer_status['customer-age-days'] = (
        df['order-date'].max() - customer_status['first-purchase']
    ).dt.days

    customer_status['purchase-frequency'] = (
        customer_status['total_orders'] /
        customer_status['customer-age-days'].clip(lower=1)
    )

    return customer_status

@st.cache_data
def call_churn(df):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "churn_predictor.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    features = pickle.load(open(os.path.join(ML_DIR, 'churn_features.pkl'), 'rb'))
    df['order-date'] = pd.to_datetime(df['order-date'])

    test_data = data_prep(df)
    features = [
        'recency',
        'total_orders',
        'total_revenue',
        'average_order_value',     
        'return_rate',
        'cancelled_rate',          
        'customer-age-days',       
        'purchase-frequency'       
    ]   
    x = test_data[features].fillna(0)
    churn_probabilities = model.predict_proba(x)[:,1]

    # adding to the dataframe
    test_data['probabilities'] = churn_probabilities

    #pd.cut will divide the data into particular segments

    test_data['risk segment'] = pd.cut(
        test_data['probabilities'],
        bins = [0,0.3,0.7,1.0],
        labels=['🟢 Stable', '🟡 At Risk', '🔴 Critical']
    )
    return test_data

def executive_kpi_section(df):
    col1,col2,col3,col4 = st.columns(4)

@st.cache_data
def prepare_data_for_prophet(df):

    raw_data = df[~df['status'].isin(['Cancelled','Returned'])]

    prophet_df = (
        raw_data
        .groupby('order-date')['total-value']
        .sum()
        .reset_index()
        .rename(columns={'order-date': 'ds', 'total-value': 'y'})
        .sort_values('ds')
        .reset_index(drop=True)
    )

    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

    return prophet_df


@st.cache_data
def call_prophet(df, forecast_days=30):

    data = prepare_data_for_prophet(df)

    # Train fresh — avoids pkl timestamp conflict
    model = Prophet(
        yearly_seasonality      = True,
        weekly_seasonality      = True,
        daily_seasonality       = False,
        seasonality_mode        = 'multiplicative',
        changepoint_prior_scale = 0.05,
        seasonality_prior_scale = 10,
        interval_width          = 0.95
    )
    model.add_seasonality(name='festive_season', period=365.25, fourier_order=5)
    model.fit(data)

    # Create future dates manually
    last_date    = data['ds'].max()
    future_dates = pd.date_range(
        start = data['ds'].min(),
        end   = last_date + pd.Timedelta(days=forecast_days),
        freq  = 'D'
    )
    future   = pd.DataFrame({'ds': future_dates})
    forecast = model.predict(future)

    return forecast
