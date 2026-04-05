import pandas as pd
import numpy as np
import mitosheet
import math

# logic for cleaning ,handeling nulls, and math logic

def handle_null_data(datas):
    # handeling nulls
    total = datas.isnull().sum().sum()
    size = datas.size
    if(total<=0.3*size):
        datas = datas.dropna()
    else:
        datas = datas.ffill().fillna(0)

    return datas

def handle_duplicate_data(datas):
    # handeling duplicates
    datas.drop_duplicates()
    return datas

def string_to_numeric_data(datas):
    # handeling string to numeric
    column = datas.columns.tolist()
    for obj in column:
        datas[obj] = pd.to_numeric(datas[obj],errors = 'ignore')

    return datas

def removal_of_outliers(datas,strategy = 'cap'):
    # handeling outliers
    column = datas.select_dtypes(include=['number']).columns
    for obj in column:
        Q1 = datas[obj].quantile(0.25)
        Q3 = datas[obj].quantile(0.75)
        IQR = Q3-Q1

        lower_fence = Q1 - 3.0*IQR
        upper_fence = Q3 + 3.0*IQR

        if strategy == 'drop':
            datas = datas[(datas[obj]>=lower_fence)&(datas[obj]<=upper_fence)]
        elif strategy == 'cap':
            datas[obj] = datas[obj].clip(lower=lower_fence, upper=upper_fence)

    return datas

def detect_anomalies(df):
    Q1 = df['total-value'].quantile(0.25)
    Q3 = df['total-value'].quantile(0.75)
    IQR = Q3-Q1
    upperbound = Q3+1.5*IQR

    anamolies = df[df['total-value']>upperbound]
    return len(anamolies)









