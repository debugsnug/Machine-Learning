"""
Comprehensive analysis for LAB 10: A1-A5
- correlation heatmap
- PCA (99% and 95%) + modeling
- Sequential feature selection
- Explainability with LIME and SHAP

Produces figures in ./results/
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Local import of helper functions
from Codes import load_and_preprocess_data, correlation_analysis, pca_analysis_99, pca_analysis_95, sequential_feature_selection, compare_results, train_and_evaluate_models, explainability_analysis

RESULTS_DIR = Path(__file__).parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)
os.chdir(RESULTS_DIR)


def main():
    # Load data using the helper
    X, y, feature_names, le = load_and_preprocess_data(None)

    # A1: Correlation heatmap (saves into current directory)
    corr = correlation_analysis(X, feature_names)

    # A2: PCA 99%
    results_pca99, pca99, scaler99, X_train_pca99, X_test_pca99, y_train, y_test = pca_analysis_99(X, y, feature_names)

    # A3: PCA 95%
    results_pca95, pca95, scaler95 = pca_analysis_95(X, y, feature_names)

    # A4: Sequential Feature Selection
    results_sfs, sfs, selected_features = sequential_feature_selection(X, y, feature_names)

    # A4.1 Save selected features
    pd.Series(selected_features).to_csv('sfs_selected_features.csv', index=False)

    # Compare
    comparison_df = compare_results(results_pca99, results_pca95, results_sfs)
    comparison_df.to_csv('model_comparison_results.csv', index=False)

    # A5: Explainability (use RandomForest from PCA99 results)
    best_model = results_pca99['Random Forest']['model']
    explainability_analysis(
        X_train_pca99, X_test_pca99, y_train, y_test,
        best_model,
        [f'PC{i+1}' for i in range(X_train_pca99.shape[1])],
        le.classes_
    )

    print('\nAll results written to', RESULTS_DIR)

if __name__ == '__main__':
    main()
