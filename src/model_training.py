# src/model_training.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import joblib
import os

def train_baseline_model(X_train, y_train):
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    return lr

def train_advanced_models(X_train, y_train):
    models = {}

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf

    # LightGBM with pipeline
    lgbm = LGBMClassifier(
        random_state=42,
        verbosity=-1,
        is_unbalance=True  # ← Helps with class imbalance
    )

    param_dist = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(3, 12),
        'learning_rate': uniform(0.01, 0.2),
        'subsample': uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.7, 0.3),
    }

    random_search = RandomizedSearchCV(
        lgbm, param_distributions=param_dist,
        n_iter=30, cv=3, scoring='roc_auc',
        random_state=42, n_jobs=-1, verbose=0
    )

    random_search.fit(X_train, y_train)
    models['LightGBM'] = random_search.best_estimator_

    return models

def save_model(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)