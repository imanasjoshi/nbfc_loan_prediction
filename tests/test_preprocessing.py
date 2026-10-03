import pandas as pd
from src.feature_engineering import create_features

def test_create_features():
    df = pd.DataFrame({
        'total_revolving_limit': [10000, 0],
        'revolving_balance': [5000, 5000],
        'annual_income': [100000, 0],
        'loan_amnt': [20000, 10000],
        'total_current_balance': [50000, 10000],
        'loan_term': ['3 years', '5 years']
    })
    
    df_fe = create_features(df)
    
    assert df_fe['credit_utilization'][0] == 0.5
    assert df_fe['credit_utilization'][1] == 0.0
    
    assert df_fe['loan_to_income'][0] == 0.2
    assert df_fe['loan_to_income'][1] == 0.0
    
    assert df_fe['loan_term_num'][0] == 3
    assert df_fe['loan_term_num'][1] == 5
