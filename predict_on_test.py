# predict_on_test.py

import pandas as pd
import joblib
from src.data_preprocessing import load_and_clean_data, encode_categorical_features, handle_outliers
from src.data_preprocessing import select_priority_features
import os

def main():
    print("🔮 Loading trained model...")
    model = joblib.load('models/best_model.pkl')

    print("📂 Loading test data...")
    df_test = load_and_clean_data('data/test.csv')
    customer_ids = df_test.index.copy()

    df_test = encode_categorical_features(df_test)
    df_test = handle_outliers(df_test, df_test.columns)

    # Select same priority features
    # Create dummy labels for compatibility
    df_test_with_labels = df_test.copy()
    df_test_with_labels['labels'] = 0
    df_test_priority = select_priority_features(df_test_with_labels, 'labels')
    X_test = df_test_priority.drop(['labels'], axis=1)

    print("📈 Predicting churn probabilities...")
    y_proba = model.predict_proba(X_test)[:, 1]

    submission = pd.DataFrame({
        'labels': y_proba
    })

    os.makedirs('output', exist_ok=True)
    submission.to_csv('output/submission.csv', index=False)
    print(f"✅ Submission saved: {len(submission)} rows")

    if os.path.exists('data/sample_submission.xlsx'):
        sample = pd.read_excel('data/sample_submission.xlsx')
        assert len(submission) == len(sample), "Length mismatch!"
        print("✅ Format validated against sample_submission.xlsx")

if __name__ == "__main__":
    main()