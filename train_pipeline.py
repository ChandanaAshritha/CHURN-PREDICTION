# train_pipeline.py

from src.utils import setup_project_directories
from src.data_preprocessing import load_and_clean_data, encode_categorical_features, handle_outliers, prepare_train_test
from src.model_training import train_baseline_model, train_advanced_models, save_model
from src.model_evaluation import evaluate_model
from src.model_interpretation import explain_with_shap
import pandas as pd

def main():
    print("🚀 Starting Churn Prediction Training Pipeline...")

    setup_project_directories()

    df = load_and_clean_data('data/train.csv')
    df = encode_categorical_features(df)
    df = handle_outliers(df, df.columns[:-1])

    X_train_res, X_val, y_train_res, y_val = prepare_train_test(df, target_col='labels')

    print("🧠 Training models on priority features...")
    baseline_model = train_baseline_model(X_train_res, y_train_res)
    advanced_models = train_advanced_models(X_train_res, y_train_res)

    all_metrics = []
    best_model = None
    best_score = 0
    best_name = ""

    print("📊 Evaluating models with comprehensive metrics...")
    
    baseline_metrics = evaluate_model(baseline_model, X_val, y_val, "Logistic Regression")
    all_metrics.append(baseline_metrics)
    print(f"Logistic Regression - AUC: {baseline_metrics['AUC']:.4f}, Precision: {baseline_metrics['Precision']:.4f}, Recall: {baseline_metrics['Recall']:.4f}, F1: {baseline_metrics['F1-Score']:.4f}, Composite: {baseline_metrics['Composite_Score']:.4f}")

    for name, model in advanced_models.items():
        metrics = evaluate_model(model, X_val, y_val, name)
        all_metrics.append(metrics)
        print(f"{name} - AUC: {metrics['AUC']:.4f}, Precision: {metrics['Precision']:.4f}, Recall: {metrics['Recall']:.4f}, F1: {metrics['F1-Score']:.4f}, Composite: {metrics['Composite_Score']:.4f}")
        
        # Use composite score for model selection instead of just AUC
        if metrics['Composite_Score'] > best_score:
            best_score = metrics['Composite_Score']
            best_model = model
            best_name = name

    save_model(best_model, 'models/best_model.pkl')
    print(f"\n🏆 Best Model: {best_name}")
    print(f"   Composite Score: {best_score:.4f}")
    print(f"   AUC: {[m for m in all_metrics if m['Model'] == best_name][0]['AUC']:.4f}")
    print(f"   Precision: {[m for m in all_metrics if m['Model'] == best_name][0]['Precision']:.4f}")
    print(f"   Recall: {[m for m in all_metrics if m['Model'] == best_name][0]['Recall']:.4f}")
    print(f"   F1-Score: {[m for m in all_metrics if m['Model'] == best_name][0]['F1-Score']:.4f}")

    if best_name in ['LightGBM', 'Random Forest']:
        explain_with_shap(best_model, X_val, best_name)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv('reports/performance_metrics.csv', index=False)
    print("✅ Metrics saved to reports/performance_metrics.csv")
    
    # Display comparison table
    print("\n📋 Model Comparison Summary:")
    print("=" * 80)
    for metrics in all_metrics:
        print(f"{metrics['Model']:20} | AUC: {metrics['AUC']:.3f} | Prec: {metrics['Precision']:.3f} | Rec: {metrics['Recall']:.3f} | F1: {metrics['F1-Score']:.3f} | Composite: {metrics['Composite_Score']:.3f}")
    print("=" * 80)
    print("🎉 Training pipeline completed!")

if __name__ == "__main__":
    main()