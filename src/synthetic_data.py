# src/synthetic_data.py

import pandas as pd
import numpy as np

def generate_synthetic_data(n_samples=10000, random_state=42):
    """
    Generate synthetic dataset matching your structure.
    Useful for testing, demos, or data augmentation.
    """
    np.random.seed(random_state)

    # Mimic your 16 features + labels
    synthetic_data = pd.DataFrame({
        'feature_0': np.random.normal(0, 1, n_samples),
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(0, 1, n_samples),
        'feature_3': np.random.normal(0, 1, n_samples),
        'feature_4': np.random.normal(0, 1, n_samples),
        'feature_5': np.random.normal(0, 1, n_samples),
        'feature_6': np.random.normal(0, 1, n_samples),  # Often top predictor
        'feature_7': np.random.randint(0, 6, n_samples),
        'feature_8': np.random.randint(0, 3, n_samples),
        'feature_9': np.random.randint(0, 4, n_samples),
        'feature_10': np.random.randint(0, 2, n_samples),
        'feature_11': np.random.randint(0, 2, n_samples),
        'feature_12': np.random.randint(0, 2, n_samples),
        'feature_13': np.random.randint(0, 3, n_samples),
        'feature_14': np.random.randint(1, 12, n_samples),  # tenure
        'feature_15': np.random.randint(0, 4, n_samples),   # claim count
    })

    # Generate synthetic labels based on feature_6 (risk_score)
    prob_churn = 1 / (1 + np.exp(-2 * synthetic_data['feature_6']))
    synthetic_data['labels'] = np.random.binomial(1, prob_churn)

    return synthetic_data

if __name__ == "__main__":
    df_synthetic = generate_synthetic_data()
    df_synthetic.to_csv('../data/synthetic_train.csv', index=False)
    print("✅ Synthetic data saved to data/synthetic_train.csv")
    print(df_synthetic['labels'].value_counts(normalize=True))