import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from .feature_mapping import FEATURE_MAPPING

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df = df.fillna(df.median(numeric_only=True))
    return df

def encode_categorical_features(df):
    return df

def handle_outliers(df, columns):
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = np.clip(df[col], lower_bound, upper_bound)
    return df

def select_priority_features(df, target_col='labels'):
    if target_col not in df.columns:
        target_col = df.columns[-1]
    corr = df.corr()[target_col].abs().sort_values(ascending=False)
    priority_features = corr.index[1:9].tolist()
    print(f"✅ Priority Features: {priority_features}")
    return df[priority_features + [target_col]]

def plot_feature_correlation_heatmap(df, reports_dir='reports/eda_plots'):
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title("Feature Correlation Heatmap (Priority Features)", fontsize=16)
    os.makedirs(reports_dir, exist_ok=True)
    plt.savefig(f'{reports_dir}/feature_correlation_heatmap.png', bbox_inches='tight', dpi=150)
    plt.close()

def prepare_train_test(df, target_col='labels', test_size=0.2, random_state=42):
    if target_col not in df.columns:
        target_col = df.columns[-1]
    df_priority = select_priority_features(df, target_col)
    plot_feature_correlation_heatmap(df_priority)
    X = df_priority.drop([target_col], axis=1, errors='ignore')
    y = df_priority[target_col]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    return X_train_res, X_val, y_train_res, y_val