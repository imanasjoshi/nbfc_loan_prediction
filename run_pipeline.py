import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from catboost import CatBoostClassifier

import src.config as config
from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import create_features

def evaluate_model(y_true, y_pred, y_prob):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'pr_auc': average_precision_score(y_true, y_prob),
        'gini': 2 * roc_auc_score(y_true, y_prob) - 1
    }
    return metrics

def train_origination_model(X, y, X_test_sub, test_ids, cat_cols):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE)
    
    # Dummy
    dummy = DummyClassifier(strategy='prior')
    dummy.fit(X_train, y_train)
    
    # LogReg Pipeline
    num_cols = [c for c in X.columns if c not in cat_cols]
    
    # CatBoost
    cat_features_idx = [i for i, col in enumerate(X.columns) if col in cat_cols]
    
    cb = CatBoostClassifier(
        iterations=300, 
        learning_rate=0.05, 
        depth=6, 
        cat_features=cat_features_idx,
        eval_metric='AUC',
        random_seed=config.RANDOM_STATE,
        verbose=50,
        auto_class_weights='Balanced'
    )
    
    cb.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=30)
    
    # Eval
    y_prob_cb = cb.predict_proba(X_val)[:, 1]
    y_pred_cb = (y_prob_cb > 0.5).astype(int)
    
    metrics = evaluate_model(y_val, y_pred_cb, y_prob_cb)
    print("Origination Model Metrics:", metrics)
    
    # Save model
    cb.save_model(str(config.MODEL_DIR / "origination_model.cbm"))
    
    # Save metrics
    with open(config.METRICS_DIR / "origination_metrics.json", "w") as f:
        json.dump(metrics, f)
        
    # Threshold Analysis
    thresholds = np.arange(0.1, 0.95, 0.05)
    thresh_results = []
    for t in thresholds:
        pred_t = (y_prob_cb > t).astype(int)
        thresh_results.append({
            'threshold': round(t, 2),
            'precision': precision_score(y_val, pred_t, zero_division=0),
            'recall': recall_score(y_val, pred_t, zero_division=0),
            'f1': f1_score(y_val, pred_t, zero_division=0)
        })
    pd.DataFrame(thresh_results).to_csv(config.METRICS_DIR / "origination_thresholds.csv", index=False)
        
    # SHAP
    if HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(cb)
            shap_values = explainer.shap_values(X_val)
        except Exception as e:
            print("SHAP failed:", e)
    else:
        print("SHAP not available, skipping.")
    
    # Make Submission
    test_probs = cb.predict_proba(X_test_sub)[:, 1]
    test_preds = (test_probs > 0.5).astype(int)
    sub = pd.DataFrame({'ID': test_ids, 'default': test_preds})
    sub.to_csv(config.ROOT_DIR / "submission.csv", index=False)
    
    # Feature importance
    feat_imp = pd.DataFrame({'feature': X.columns, 'importance': cb.get_feature_importance()})
    feat_imp = feat_imp.sort_values('importance', ascending=False)
    feat_imp.to_csv(config.METRICS_DIR / "origination_feature_importance.csv", index=False)

    return cb

def train_monitoring_model(X, y, cat_cols):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE)
    
    cat_features_idx = [i for i, col in enumerate(X.columns) if col in cat_cols]
    
    cb = CatBoostClassifier(
        iterations=300, 
        learning_rate=0.05, 
        depth=6, 
        cat_features=cat_features_idx,
        eval_metric='AUC',
        random_seed=config.RANDOM_STATE,
        verbose=50,
        auto_class_weights='Balanced'
    )
    
    cb.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=30)
    
    y_prob_cb = cb.predict_proba(X_val)[:, 1]
    y_pred_cb = (y_prob_cb > 0.5).astype(int)
    
    metrics = evaluate_model(y_val, y_pred_cb, y_prob_cb)
    print("Monitoring Model Metrics:", metrics)
    
    cb.save_model(str(config.MODEL_DIR / "monitoring_model.cbm"))
    
    with open(config.METRICS_DIR / "monitoring_metrics.json", "w") as f:
        json.dump(metrics, f)

    return cb

def main():
    print("Loading data...")
    train, test = load_data()
    
    print("Cleaning data...")
    train_clean = clean_data(train)
    test_clean = clean_data(test)
    
    print("Feature engineering...")
    train_fe = create_features(train_clean)
    test_fe = create_features(test_clean)
    
    # Target
    y = train_fe[config.TARGET]
    
    # Prepare Origination Data
    orig_features = config.ORIGINATION_FEATURES + ['credit_utilization', 'loan_to_income', 'balance_to_income', 'revolving_to_income', 'loan_term_num']
    X_orig = train_fe[orig_features]
    X_test_orig = test_fe[orig_features]
    cat_cols_orig = config.CAT_COLS
    
    print("Training Origination Model...")
    cb_orig = train_origination_model(X_orig, y, X_test_orig, test_fe['ID'], cat_cols_orig)
    
    # Prepare Monitoring Data
    mon_features = orig_features + config.LEAKAGE_COLS
    X_mon = train_fe[mon_features]
    cat_cols_mon = config.CAT_COLS
    
    print("Training Monitoring Model...")
    cb_mon = train_monitoring_model(X_mon, y, cat_cols_mon)
    
    # Save expected columns
    meta = {
        'origination_features': orig_features,
        'monitoring_features': mon_features,
        'cat_cols': config.CAT_COLS
    }
    with open(config.MODEL_DIR / "metadata.json", "w") as f:
        json.dump(meta, f)
        
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
