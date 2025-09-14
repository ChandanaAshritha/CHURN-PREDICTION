# src/model_interpretation.py

import shap
import matplotlib.pyplot as plt
import os

def explain_with_shap(model, X_test, model_name, reports_dir='reports'):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Handle different shap_values formats
    if isinstance(shap_values, list):
        # For binary classification, use the positive class values
        shap_values_matrix = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_values_matrix = shap_values

    # Create reports directory
    os.makedirs(f'{reports_dir}/shap_plots', exist_ok=True)

    # Feature Importance (Bar)
    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values_matrix, X_test, plot_type="bar", show=False)
    plt.title(f'{model_name} - Global Feature Importance (Priority Features)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{reports_dir}/shap_plots/{model_name}_feature_importance.png', bbox_inches='tight')
    plt.close()

    # Beeswarm
    plt.figure(figsize=(10,8))
    shap.summary_plot(shap_values_matrix, X_test, show=False)
    plt.title(f'{model_name} - SHAP Beeswarm Plot (Priority Features)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{reports_dir}/shap_plots/{model_name}_beeswarm.png', bbox_inches='tight')
    plt.close()

    return explainer, shap_values