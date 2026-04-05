import pandas as pd
import numpy as np
from utils import handle_null_data,handle_duplicate_data,string_to_numeric_data,removal_of_outliers


def run_clean_pipeline(raw_df):

    datas = handle_null_data(raw_df)
    datas = handle_duplicate_data(datas)
    datas = string_to_numeric_data(datas)
    datas = removal_of_outliers(datas,strategy='cap')

    return datas

