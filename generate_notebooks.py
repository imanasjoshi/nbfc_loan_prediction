import nbformat as nbf
import os

os.makedirs('notebooks', exist_ok=True)

# 01_data_understanding_eda.ipynb
nb1 = nbf.v4.new_notebook()
nb1.cells = [
    nbf.v4.new_markdown_cell("# Data Understanding and EDA\n\nThis notebook performs data quality analysis and exploratory data analysis."),
    nbf.v4.new_code_cell("import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings('ignore')"),
    nbf.v4.new_code_cell("train = pd.read_csv('../data/raw/Train.csv')\ntest = pd.read_csv('../data/raw/Test.csv')\ntrain.head()"),
    nbf.v4.new_code_cell("train.info()"),
    nbf.v4.new_code_cell("train['default'].value_counts(normalize=True).plot(kind='bar', title='Default Rate')"),
    nbf.v4.new_code_cell("sns.boxplot(x='default', y='loan_amnt', data=train)\nplt.title('Loan Amount by Default Status')"),
    nbf.v4.new_code_cell("sns.boxplot(x='default', y='interest_rate', data=train)\nplt.title('Interest Rate by Default Status')")
]
with open('notebooks/01_data_understanding_eda.ipynb', 'w') as f:
    nbf.write(nb1, f)

# 02_feature_engineering.ipynb
nb2 = nbf.v4.new_notebook()
nb2.cells = [
    nbf.v4.new_markdown_cell("# Feature Engineering\n\nCreating new features from the raw data."),
    nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\ntrain = pd.read_csv('../data/raw/Train.csv')"),
    nbf.v4.new_code_cell("train['credit_utilization'] = np.where(train['total_revolving_limit'] > 0, train['revolving_balance'] / train['total_revolving_limit'], 0)"),
    nbf.v4.new_code_cell("train['loan_to_income'] = np.where(train['annual_income'] > 0, train['loan_amnt'] / train['annual_income'], 0)"),
    nbf.v4.new_code_cell("train[['credit_utilization', 'loan_to_income']].describe()")
]
with open('notebooks/02_feature_engineering.ipynb', 'w') as f:
    nbf.write(nb2, f)

# 03_model_development.ipynb
nb3 = nbf.v4.new_notebook()
nb3.cells = [
    nbf.v4.new_markdown_cell("# Model Development\n\nTraining baseline and advanced models."),
    nbf.v4.new_code_cell("import sys\nsys.path.append('..')\nfrom src.data_preprocessing import load_data, clean_data\nfrom src.feature_engineering import create_features\nimport src.config as config\nfrom catboost import CatBoostClassifier\nfrom sklearn.model_selection import train_test_split"),
    nbf.v4.new_code_cell("train, test = load_data()\ntrain_c = clean_data(train)\ntrain_f = create_features(train_c)"),
    nbf.v4.new_code_cell("X = train_f[config.ORIGINATION_FEATURES]\ny = train_f[config.TARGET]"),
    nbf.v4.new_code_cell("X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)"),
    nbf.v4.new_code_cell("# Train model\ncat_idx = [i for i, c in enumerate(X.columns) if c in config.CAT_COLS]\ncb = CatBoostClassifier(iterations=100, cat_features=cat_idx, verbose=50)\ncb.fit(X_train, y_train, eval_set=(X_val, y_val))")
]
with open('notebooks/03_model_development.ipynb', 'w') as f:
    nbf.write(nb3, f)

# 04_model_explainability.ipynb
nb4 = nbf.v4.new_notebook()
nb4.cells = [
    nbf.v4.new_markdown_cell("# Model Explainability\n\nUsing SHAP to explain model predictions."),
    nbf.v4.new_code_cell("import shap\nimport joblib\nimport pandas as pd\nfrom catboost import CatBoostClassifier"),
    nbf.v4.new_code_cell("# Load model (assuming trained model is saved)\n# cb = CatBoostClassifier().load_model('../models/origination_model.cbm')"),
    nbf.v4.new_code_cell("# explainer = shap.TreeExplainer(cb)\n# shap_values = explainer.shap_values(X_val)"),
    nbf.v4.new_code_cell("# shap.summary_plot(shap_values, X_val)")
]
with open('notebooks/04_model_explainability.ipynb', 'w') as f:
    nbf.write(nb4, f)

# 05_business_analysis.ipynb
nb5 = nbf.v4.new_notebook()
nb5.cells = [
    nbf.v4.new_markdown_cell("# Business Analysis\n\nExtracting business insights and risk segments."),
    nbf.v4.new_code_cell("import pandas as pd\ntrain = pd.read_csv('../data/raw/Train.csv')"),
    nbf.v4.new_code_cell("train.groupby('loan_grade')['default'].mean().plot(kind='bar')"),
    nbf.v4.new_code_cell("train.groupby('loan_purpose')['default'].mean().sort_values().plot(kind='barh')")
]
with open('notebooks/05_business_analysis.ipynb', 'w') as f:
    nbf.write(nb5, f)

print("Notebooks generated.")
