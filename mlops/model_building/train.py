
# for data manipulation
import pandas as pd
import os
import joblib

# preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# model training
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Hugging Face
from huggingface_hub import (
    login,
    HfApi,
    create_repo,
    hf_hub_download
)
from huggingface_hub.utils import RepositoryNotFoundError

# --------------------------------------------------
# Hugging Face Authentication
# --------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found")

login(token=HF_TOKEN)

print("Hugging Face login successful")

# --------------------------------------------------
# Download dataset from HF Dataset Repo
# --------------------------------------------------

dataset_repo_id = "jadhavsainath/bank-customer-churn"

Xtrain_path = hf_hub_download(
    repo_id=dataset_repo_id,
    filename="Xtrain.csv",
    repo_type="dataset"
)

Xtest_path = hf_hub_download(
    repo_id=dataset_repo_id,
    filename="Xtest.csv",
    repo_type="dataset"
)

ytrain_path = hf_hub_download(
    repo_id=dataset_repo_id,
    filename="ytrain.csv",
    repo_type="dataset"
)

ytest_path = hf_hub_download(
    repo_id=dataset_repo_id,
    filename="ytest.csv",
    repo_type="dataset"
)

print("Dataset files downloaded successfully")

# --------------------------------------------------
# Read data
# --------------------------------------------------

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)

ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# Convert dataframe -> series
ytrain = ytrain.iloc[:, 0]
ytest = ytest.iloc[:, 0]

print("Dataset loaded successfully")
print("Xtrain shape:", Xtrain.shape)
print("Xtest shape:", Xtest.shape)

# --------------------------------------------------
# Features
# --------------------------------------------------

numeric_features = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary"
]

categorical_features = [
    "Geography"
]

# --------------------------------------------------
# Handle Class Imbalance
# --------------------------------------------------

class_weight = (
    ytrain.value_counts()[0]
    / ytrain.value_counts()[1]
)

print("Scale Pos Weight:", class_weight)

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# --------------------------------------------------
# Model
# --------------------------------------------------

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss"
)

# --------------------------------------------------
# Hyperparameter Grid
# --------------------------------------------------

param_grid = {
    "xgbclassifier__n_estimators": [50, 100],
    "xgbclassifier__max_depth": [3, 4],
    "xgbclassifier__learning_rate": [0.05, 0.1]
}

# --------------------------------------------------
# Pipeline
# --------------------------------------------------

model_pipeline = make_pipeline(
    preprocessor,
    xgb_model
)

# --------------------------------------------------
# Training
# --------------------------------------------------

grid_search = GridSearchCV(
    model_pipeline,
    param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(Xtrain, ytrain)

print("Best Parameters:")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_

# --------------------------------------------------
# Evaluation
# --------------------------------------------------

classification_threshold = 0.45

y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

print("\nTraining Report")
print(classification_report(ytrain, y_pred_train))

print("\nTesting Report")
print(classification_report(ytest, y_pred_test))

# --------------------------------------------------
# Save Model
# --------------------------------------------------

model_file = "best_churn_model.joblib"

joblib.dump(best_model, model_file)

print("Model saved successfully")

# --------------------------------------------------
# Upload Model to Hugging Face
# --------------------------------------------------

repo_id = "jadhavsainath/churn-model"
repo_type = "model"

api = HfApi(token=HF_TOKEN)

try:
    api.repo_info(
        repo_id=repo_id,
        repo_type=repo_type
    )
    print(f"Model repo '{repo_id}' already exists")

except RepositoryNotFoundError:

    print(f"Creating model repo '{repo_id}'")

    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False
    )

api.upload_file(
    path_or_fileobj=model_file,
    path_in_repo=model_file,
    repo_id=repo_id,
    repo_type=repo_type
)

print("Model uploaded successfully")
