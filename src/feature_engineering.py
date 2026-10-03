import pandas as pd
import numpy as np

def create_features(df):
    df = df.copy()
    
    # credit_utilization
    df['credit_utilization'] = np.where(df['total_revolving_limit'] > 0, 
                                        df['revolving_balance'] / df['total_revolving_limit'], 0)
    
    # loan_to_income
    df['loan_to_income'] = np.where(df['annual_income'] > 0, 
                                    df['loan_amnt'] / df['annual_income'], 0)
    
    # balance_to_income
    df['balance_to_income'] = np.where(df['annual_income'] > 0, 
                                       df['total_current_balance'] / df['annual_income'], 0)
                                       
    # revolving_to_income
    df['revolving_to_income'] = np.where(df['annual_income'] > 0, 
                                         df['revolving_balance'] / df['annual_income'], 0)
                                         
    # Convert loan_term to numerical (e.g. '3 years' -> 3, '5 years' -> 5)
    if 'loan_term' in df.columns:
        df['loan_term_num'] = df['loan_term'].astype(str).str.extract(r'(\d+)').astype(float)
        df['loan_term_num'] = df['loan_term_num'].fillna(0)
        
    return df
