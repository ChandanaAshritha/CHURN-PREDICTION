# src/feature_stats.py

import pandas as pd
from .feature_mapping import FEATURE_MAPPING

def generate_feature_stats():
    df = pd.read_csv('../data/Train.csv')
    df = df.rename(columns=FEATURE_MAPPING)  # Apply mapping
    churned = df[df['labels'] == 1]
    retained = df[df['labels'] == 0]

    print("=== FEATURE STATISTICS: CHURNED vs RETAINED ===")
    for col in FEATURE_MAPPING.values():
        if col in df.columns and col != 'labels':
            churned_mean = churned[col].mean()
            retained_mean = retained[col].mean()
            diff = churned_mean - retained_mean
            print(f"{col:25}: Churned={churned_mean:>7.3f}, Retained={retained_mean:>7.3f}, Diff={diff:>7.3f}")

if __name__ == "__main__":

    generate_feature_stats()
