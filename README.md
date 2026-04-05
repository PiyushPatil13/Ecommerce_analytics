# 🛒 E-Commerce Analytics Pro

A full-stack machine learning web application for ecommerce business intelligence — featuring revenue forecasting, customer churn prediction, risk analysis, and live commodity tracking.

---

## 📌 Project Overview

E-Commerce Analytics Pro is an end-to-end data analytics platform built with Python and Streamlit. It takes raw ecommerce transaction data and transforms it into actionable business insights through machine learning models, interactive dashboards, and predictive forecasts.

The project covers the complete ML pipeline — from synthetic data generation and feature engineering, all the way to a deployed multi-page web application.

---

## 🚀 Features

### 📊 Standard Dashboard
- Revenue KPI cards — total revenue, average order value, order count
- Sales trend analysis with interactive Plotly charts
- Customer analytics and segmentation
- Anomaly detection on transactions

### 🔍 Strategic Insights
- Multi-dimensional risk analysis across 6 categories
- Product category performance breakdown
- City-level revenue distribution
- Order status analysis (Delivered, Returned, Cancelled)

### 🤖 Predictive Insights
- **Customer Churn Prediction** — classifies customers into 3 risk segments
  - 🟢 Stable
  - 🟡 At Risk
  - 🔴 Critical
- Churn probability scores with color-coded risk table
- Downloadable high-risk and moderate-risk customer reports
- **Revenue Forecasting with Prophet** — 30–90 day future forecast
  - Actual vs forecast chart with 95% confidence band
  - Trend, weekly, and yearly seasonality components
  - Interactive forecast table with daily predictions

### 📈 Commodity Tracker
- Live commodity price tracking via Yahoo Finance
- Supports Electronics, Food & Grocery, Clothing & Textiles
- Normalized price comparison (Base = 100)
- Configurable period (1mo, 3mo, 6mo, 1y) and interval (daily, weekly)

---

## 🧠 Machine Learning Models

| Model | Task | Performance |
|---|---|---|
| XGBoost | Revenue forecasting | ~2.68% MAPE |
| Facebook Prophet | Revenue forecasting + decomposition | Trend + seasonality |
| Custom Churn Model | Customer risk classification | 3-class segmentation |

### Feature Engineering (for XGBoost)
- **Lag features** — revenue 1, 2, 3, 7, 14, 21, 30 days ago
- **Rolling averages** — 7, 14, 30 day windows
- **Rolling statistics** — std deviation, min, max
- **Calendar features** — day of week, month, quarter, is_weekend, is_festive
- **Trend features** — week-over-week change, momentum
- **Category features** — per-category daily revenue breakdown
---

## ⚙️ Installation

### Prerequisites
- Python 3.9+
- Anaconda (recommended for Prophet on Windows)

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ecommerce-analytics-pro.git
cd ecommerce-analytics-pro
```

### 2. Install Prophet (via conda — recommended on Windows)
```bash
conda install -c conda-forge prophet
```

### 3. Install remaining dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify installation
```bash
python -c "from prophet import Prophet; import xgboost; import streamlit; print('All packages OK')"
```

---

## 📦 Requirements

```
streamlit
pandas
numpy
plotly
xgboost
scikit-learn
prophet
statsmodels
matplotlib
faker
yfinance
```

---

## ▶️ Running the App

```bash
cd src
streamlit run app.py
```

The app will open at ``

---

## 📁 Data Format

Upload a CSV file with the following columns:

| Column | Type | Description |
|---|---|---|
| `order-id` | string | Unique order identifier |
| `customer-name` | string | Customer full name |
| `customer-email` | string | Customer email |
| `order-date` | date | Order date (YYYY-MM-DD) |
| `product-category` | string | Electronics / Grocery / Clothing / Decorative |
| `price` | float | Unit price |
| `quantity` | int | Items ordered |
| `total-value` | float | price × quantity |
| `status` | string | Delivered / On-way / Cancelled / Returned |
| `city` | string | Customer city |

### Generate Sample Data
```bash
python scripts/generate_revenue_data.py
```
This generates `Sample4_data.csv` with 65,000 realistic synthetic rows including seasonality, weekend effects, and festive patterns.

---

## 🔧 Training Your Own Models

### XGBoost
```bash
python scripts/feature_engineering.py    # creates featured_revenue.csv
python scripts/xgboost_forecast.py       # trains and saves model
```

### Prophet
```bash
python scripts/prophet_forecast.py       # trains and saves prophet_revenue.pkl
```

---

## 📊 Model Performance

### XGBoost Revenue Forecaster
```
Train MAPE : 0.46%
Test MAPE  : 2.68%
Test R²    : 0.9881
Features   : 33
```

### Prophet Revenue Forecaster
- Learns trend, weekly seasonality (Mon-Sun patterns), and yearly seasonality
- Best suited for real multi-year data
- Outputs confidence intervals for uncertainty estimation

---

## 🖥️ Screenshots

| Dashboard | Predictive Insights |
|---|---|
| Revenue KPIs + Sales trends | Churn risk pie chart + forecast |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Data processing | Pandas, NumPy |
| Machine learning | XGBoost, Prophet, Scikit-learn |
| Visualisation | Plotly, Matplotlib |
| Data generation | Faker |
| Commodity data | yfinance |
| Model serialisation | Pickle |

---

## 🔮 Future Improvements

- [ ] Add SARIMA model to complete the 3-model auto-comparison
- [ ] Deploy on Streamlit Cloud for public access
- [ ] Connect to real ecommerce database (Shopify / WooCommerce API)
- [ ] Add LSTM model once 2+ years of data is available
- [ ] Add unit tests for all visualizer functions
- [ ] Add email alerts for high-risk customers

---

## 👨‍💻 Author

Built as a personal learning project to explore end-to-end machine learning, time-series forecasting, and Streamlit application development.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
