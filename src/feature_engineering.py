# src/feature_engineering.py — kept for compatibility

import pandas as pd

def create_ratio_features(df):
    # Optional — if you want to add more features later
    return df

def create_interaction_features(df):
    return df

def engineer_features(df):
    df = create_ratio_features(df)
    df = create_interaction_features(df)
    return df