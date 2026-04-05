import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix
)
import pickle
import os
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# advanced function to create the churn function

def create_churn_label(df):

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
    
    customer_status['churned_prob'] = (
        0.4 * (customer_status['recency'] / customer_status['recency'].max()) +
        0.3 * customer_status['return_rate'] +
        0.3 * (1 - customer_status['purchase-frequency'].clip(0,1))
    )

# Then generate binary churn based on this probability
    customer_status['churned'] = np.random.binomial(1, customer_status['churned_prob'])

    return customer_status

def build_features(customer_status,df):

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
        'recency',
        'total_orders',
        'total_revenue',
        'average_order_value',     
        'return_rate',
        'cancelled_rate',          
        'customer-age-days',       
        'purchase-frequency'       
    ]   

    return customer_status,features

# training function function

def train_model(customer_status,features):

    X = customer_status[features].fillna(0)
    Y = customer_status['churned']

    print(f"Total Customers : {len(X):,}")
    
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)

    # Training model
    base_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        loss_function='Logloss',
        auto_class_weights='Balanced',
        eval_metric='AUC',
        random_state=42,
        verbose=0
    )

    model = CalibratedClassifierCV(
        base_model,
        method='isotonic',
        cv=5
    )
    model.fit(X_train,Y_train)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "machine_learning")
    os.makedirs(MODEL_DIR, exist_ok=True)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    print("\n" + "="*50)
    print("MODEL PERFORMANCE")
    print("="*50)
    print(classification_report(Y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(Y_test, y_prob):.3f}")
    print("="*50)

    with open(os.path.join(MODEL_DIR, "churn_predictor.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(MODEL_DIR, "churn_features.pkl"), "wb") as f:
        pickle.dump(features, f)

    print("\n✅ Model saved successfully!")
    print("Files created:")
    print("  - churn_predictor.pkl")
    print("  - churn_features.pkl")

    return model, features


if __name__ == "__main__":
    # Load your data
    print("Loading data...")
    df = pd.read_csv('Train_data.csv')
    df['order-date'] = pd.to_datetime(df['order-date'])

    # Create labels
    print("\nCreating churn labels...")
    customer_stats = create_churn_label(df)

    # Build features
    print("Building features...")
    customer_stats, features = build_features(customer_stats, df)

    # Train
    model, features = train_model(
        customer_stats, features
    )
    print("\n🎉 Training complete!")
    print("Run your Streamlit app to see predictions.")

    







    


