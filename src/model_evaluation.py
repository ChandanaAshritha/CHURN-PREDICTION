# src/model_evaluation.py

import matplotlib
matplotlib.use('Agg')  # 👈 Critical fix for "Could not find platform independent libraries"
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc
)

def calculate_composite_score(metrics, weights=None):
    """
    Calculate a composite score based on multiple metrics.
    
    Args:
        metrics: Dictionary containing AUC, Precision, Recall, F1-Score
        weights: Dictionary with weights for each metric (default: balanced weights)
                Example: {'AUC': 0.4, 'Precision': 0.2, 'Recall': 0.2, 'F1-Score': 0.2}
    
    Returns:
        Composite score (higher is better, range 0-1)
    """
    if weights is None:
        # Balanced weights - can be customized based on business priorities
        weights = {'AUC': 0.3, 'Precision': 0.2, 'Recall': 0.2, 'F1-Score': 0.3}
    
    # Ensure weights sum to 1
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 0.001:
        weights = {k: v/total_weight for k, v in weights.items()}
    
    # Calculate weighted composite score
    composite_score = (
        weights['AUC'] * metrics['AUC'] +
        weights['Precision'] * metrics['Precision'] +
        weights['Recall'] * metrics['Recall'] +
        weights['F1-Score'] * metrics['F1-Score']
    )
    
    return composite_score

def get_model_ranking(all_metrics, weights=None):
    """
    Rank models based on composite score and return sorted list.
    
    Args:
        all_metrics: List of metric dictionaries for all models
        weights: Custom weights for composite score calculation
    
    Returns:
        List of (model_name, composite_score) tuples sorted by score (descending)
    """
    rankings = []
    for metrics in all_metrics:
        composite_score = calculate_composite_score(metrics, weights)
        rankings.append((metrics['Model'], composite_score))
    
    return sorted(rankings, key=lambda x: x[1], reverse=True)

def evaluate_model(model, X_test, y_test, model_name, reports_dir='reports'):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc_score = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        'Model': model_name,
        'AUC': auc_score,
        'Precision': report['1']['precision'],
        'Recall': report['1']['recall'],
        'F1-Score': report['1']['f1-score']
    }
    
    # Calculate composite score
    metrics['Composite_Score'] = calculate_composite_score(metrics)

    # Confusion Matrix
    plt.figure(figsize=(6,4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'{reports_dir}/eda_plots/{model_name}_confusion_matrix.png', bbox_inches='tight')
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}', color='#FF6F61', linewidth=3)
    plt.plot([0,1], [0,1], 'k--', alpha=0.5)
    plt.title(f'{model_name} - ROC Curve', fontsize=14, fontweight='bold')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{reports_dir}/eda_plots/{model_name}_roc_curve.png', bbox_inches='tight')
    plt.close()

    return metrics