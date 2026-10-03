import pandas as pd
import numpy as np
import src.config as config

def load_data():
    train = pd.read_csv(config.TRAIN_PATH)
    test = pd.read_csv(config.TEST_PATH)
    return train, test

def clean_data(df):
    df = df.copy()
    
    # Handle missing values
    # For categorical, fill with 'Missing' or mode
    for col in config.CAT_COLS:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna('Missing')
            
    # For numerical, fill with median (simple approach for baseline)
    for col in config.NUM_COLS + config.LEAKAGE_COLS:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    # Standardize loan_term to extract numbers if needed, but CatBoost can handle categories
    # Convert object columns to string just in case
    for col in config.CAT_COLS:
        df[col] = df[col].astype(str)
        
    return df
