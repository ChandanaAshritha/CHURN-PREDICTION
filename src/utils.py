import os

def ensure_directory_exists(path):
    os.makedirs(path, exist_ok=True)

def setup_project_directories():
    dirs = [
        'models',
        'reports/eda_plots',
        'reports/shap_plots',
        'output'
    ]
    for d in dirs:
        ensure_directory_exists(d)