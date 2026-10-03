import os
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "models"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
SHAP_DIR = ARTIFACTS_DIR / "shap"

TRAIN_PATH = RAW_DATA_DIR / "Train.csv"
TEST_PATH = RAW_DATA_DIR / "Test.csv"
DICT_PATH = RAW_DATA_DIR / "Data_Dictionary_.csv"

# Target and features
TARGET = "default"
ID_COL = "ID"

# Leakage variables
LEAKAGE_COLS = ["interest_receive", "last_week_pay"]

# Feature groupings
NUM_COLS = [
    "loan_amnt", "interest_rate", "annual_income", "debt_to_income", 
    "delinq_2yrs", "public_records", "revolving_balance", "total_acc", 
    "total_current_balance", "total_revolving_limit"
]

CAT_COLS = [
    "loan_term", "loan_grade", "loan_subgrade", "job_experience", 
    "home_ownership", "income_verification_status", "loan_purpose", 
    "state_code", "application_type"
]

ORIGINATION_FEATURES = NUM_COLS + CAT_COLS
MONITORING_FEATURES = ORIGINATION_FEATURES + LEAKAGE_COLS

RANDOM_STATE = 42
