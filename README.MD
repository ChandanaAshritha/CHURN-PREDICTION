# 📊 Insurance Churn Prediction System

End-to-end ML solution for predicting customer churn in insurance, built for `train.csv`/`test.csv` structure with eye-catching Streamlit UI.

## ✨ Features

- ✅ Trains on `train.csv`, predicts on `test.csv`
- ✅ Generates `submission.csv` matching `sample_submission.csv`
- ✅ Beautiful, interactive Streamlit dashboard
- ✅ SHAP explainability for business trust
- ✅ Batch CSV prediction with download
- ✅ Gauge risk visualization & retention recommendations

## 🚀 Setup & Run

```bash
# Clone and setup
git clone https://github.com/yourusername/insurance-churn-system.git
cd insurance-churn-system
pip install -r requirements.txt

# Place your files in data/
# - train.csv
# - test.csv
# - sample_submission.csv

# Train model
python train_pipeline.py

# Generate predictions
python predict_on_test.py

# Launch UI
streamlit run app.py