"""
Elephant Call Classification Analysis
Feature Selection, Dimensionality Reduction, and Explainability
"""

import pandas as pd
import numpy as np
import matplotlib
# Use non-interactive backend for headless environments (prevents tkinter errors)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# For explainability
import shap
import lime
import lime.lime_tabular
import os

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data(filepath):
    # allow filepath argument, default to the dataset in LAB 3
    file_path = filepath or r"D:\OneDrive - Amrita vishwa vidyapeetham\\SEM 5\\23CSE301 Machine Learning\\LAB\\Machine-Learning\\LAB 3\\20231225_dfall_obs_data_and_spectral_features_revision1_n469.csv"
    df = pd.read_csv(file_path)

    
    print("Dataset shape:", df.shape)
    print("\nFirst few columns:", df.columns[:10].tolist())
    print("\nTarget variable (Context2) distribution:")
    print(df['Context2'].value_counts())
    
    # Select acoustic features by explicit patterns commonly present in the file
    candidate_prefixes = ('V', 'Fw', 'Mw', 'F', 'M', 'sprs')
    feature_cols = [col for col in df.columns if col.startswith(candidate_prefixes)]
    # Remove some columns that are not numeric features if present
    feature_cols = [c for c in feature_cols if c.lower() not in ('file', 'soundfile')]
    
    # Prefer Context2 as target (care, greeting, contact). Fallback to 'Call_Type' if missing
    target_col = 'Context2' if 'Context2' in df.columns else ('Call_Type' if 'Call_Type' in df.columns else None)
    if target_col is None:
        raise ValueError('No suitable target column found (Context2 or Call_Type)')
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Remove samples with missing target
    valid_idx = ~y.isna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"\nFeatures used: {len(feature_cols)}")
    print(f"Samples: {len(X)}")
    print(f"Classes: {le.classes_}")
    
    return X, y_encoded, feature_cols, le

# ============================================================================
# A1: FEATURE CORRELATION ANALYSIS
# ============================================================================

def correlation_analysis(X, feature_names):
    """Perform correlation analysis and create heatmap"""
    print("\n" + "="*80)
    print("A1: FEATURE CORRELATION ANALYSIS")
    print("="*80)
    
    # Calculate correlation matrix
    corr_matrix = X.corr()
    
    # Plot full heatmap (may be large)
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap (All Features)', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('correlation_heatmap_full.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Find highly correlated feature pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr_pairs.append((
                    corr_matrix.columns[i], 
                    corr_matrix.columns[j], 
                    corr_matrix.iloc[i, j]
                ))
    
    print(f"\nHighly correlated feature pairs (|r| > 0.8): {len(high_corr_pairs)}")
    if high_corr_pairs:
        print("\nTop 10 highly correlated pairs:")
        for pair in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:10]:
            print(f"  {pair[0]} <-> {pair[1]}: {pair[2]:.3f}")
    
    # Plot top correlated features (subset)
    top_features = corr_matrix.abs().sum().sort_values(ascending=False).head(30).index
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix.loc[top_features, top_features], 
                annot=False, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap (Top 30 Features by Total Correlation)', 
              fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig('correlation_heatmap_top30.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nCorrelation heatmaps saved!")
    
    return corr_matrix

# ============================================================================
# MODEL TRAINING AND EVALUATION
# ============================================================================

def train_and_evaluate_models(X_train, X_test, y_train, y_test, suffix=""):
    """Train multiple models and return results"""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42, probability=True)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{name}:")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        results[name] = {
            'model': model,
            'accuracy': acc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred
        }
        
        print(f"  Test Accuracy: {acc:.4f}")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return results

# ============================================================================
# A2: PCA WITH 99% VARIANCE
# ============================================================================

def pca_analysis_99(X, y, feature_names):
    """Perform PCA with 99% variance retention"""
    print("\n" + "="*80)
    print("A2: PCA WITH 99% EXPLAINED VARIANCE")
    print("="*80)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply PCA with 99% variance
    pca = PCA(n_components=0.99, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"\nOriginal features: {X_train_scaled.shape[1]}")
    print(f"PCA components (99% variance): {pca.n_components_}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    # Plot explained variance
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(np.cumsum(pca.explained_variance_ratio_), linewidth=2)
    plt.axhline(y=0.99, color='r', linestyle='--', label='99% variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance (99%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.bar(range(1, min(21, len(pca.explained_variance_ratio_)+1)), 
            pca.explained_variance_ratio_[:20])
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Variance Explained by Top 20 Components')
    plt.tight_layout()
    plt.savefig('pca_variance_99.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Train models
    print("\nModel Performance with PCA (99% variance):")
    results_pca99 = train_and_evaluate_models(X_train_pca, X_test_pca, y_train, y_test)
    
    return results_pca99, pca, scaler, X_train_pca, X_test_pca, y_train, y_test

# ============================================================================
# A3: PCA WITH 95% VARIANCE
# ============================================================================

def pca_analysis_95(X, y, feature_names):
    """Perform PCA with 95% variance retention"""
    print("\n" + "="*80)
    print("A3: PCA WITH 95% EXPLAINED VARIANCE")
    print("="*80)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply PCA with 95% variance
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"\nOriginal features: {X_train_scaled.shape[1]}")
    print(f"PCA components (95% variance): {pca.n_components_}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    # Plot explained variance
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(np.cumsum(pca.explained_variance_ratio_), linewidth=2)
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance (95%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.bar(range(1, min(21, len(pca.explained_variance_ratio_)+1)), 
            pca.explained_variance_ratio_[:20])
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Variance Explained by Top 20 Components')
    plt.tight_layout()
    plt.savefig('pca_variance_95.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Train models
    print("\nModel Performance with PCA (95% variance):")
    results_pca95 = train_and_evaluate_models(X_train_pca, X_test_pca, y_train, y_test)
    
    return results_pca95, pca, scaler

# ============================================================================
# A4: SEQUENTIAL FEATURE SELECTION
# ============================================================================

def sequential_feature_selection(X, y, feature_names):
    """Perform sequential feature selection"""
    print("\n" + "="*80)
    print("A4: SEQUENTIAL FEATURE SELECTION")
    print("="*80)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Use Random Forest as base estimator
    base_estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    
    # Forward selection - select best 20 features
    print("\nPerforming Forward Sequential Feature Selection...")
    sfs_forward = SequentialFeatureSelector(
        base_estimator, 
        n_features_to_select=20,
        direction='forward',
        cv=3,
        n_jobs=-1
    )
    sfs_forward.fit(X_train_scaled, y_train)
    
    selected_features_idx = sfs_forward.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_features_idx]
    
    print(f"\nSelected {len(selected_features)} features:")
    for i, feat in enumerate(selected_features, 1):
        print(f"  {i}. {feat}")
    
    X_train_sfs = sfs_forward.transform(X_train_scaled)
    X_test_sfs = sfs_forward.transform(X_test_scaled)
    
    # Train models
    print("\nModel Performance with Sequential Feature Selection:")
    results_sfs = train_and_evaluate_models(X_train_sfs, X_test_sfs, y_train, y_test)
    
    return results_sfs, sfs_forward, selected_features

# ============================================================================
# COMPARISON OF ALL METHODS
# ============================================================================

def compare_results(results_pca99, results_pca95, results_sfs):
    """Compare results from all dimensionality reduction methods"""
    print("\n" + "="*80)
    print("COMPARISON OF ALL METHODS")
    print("="*80)
    
    comparison_data = []
    
    for method_name, results in [('PCA 99%', results_pca99), 
                                   ('PCA 95%', results_pca95), 
                                   ('SFS', results_sfs)]:
        for model_name, metrics in results.items():
            comparison_data.append({
                'Method': method_name,
                'Model': model_name,
                'Test Accuracy': metrics['accuracy'],
                'CV Mean': metrics['cv_mean'],
                'CV Std': metrics['cv_std']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n", comparison_df.to_string(index=False))
    
    # Visualize comparison
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    pivot_test = comparison_df.pivot(index='Model', columns='Method', values='Test Accuracy')
    pivot_test.plot(kind='bar', ax=plt.gca())
    plt.title('Test Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xlabel('Model')
    plt.legend(title='Method')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.subplot(1, 2, 2)
    pivot_cv = comparison_df.pivot(index='Model', columns='Method', values='CV Mean')
    pivot_cv.plot(kind='bar', ax=plt.gca())
    plt.title('Cross-Validation Accuracy Comparison')
    plt.ylabel('CV Accuracy')
    plt.xlabel('Model')
    plt.legend(title='Method')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('method_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nComparison plot saved!")
    
    return comparison_df

# ============================================================================
# A5: EXPLAINABILITY WITH LIME AND SHAP
# ============================================================================

def explainability_analysis(X_train, X_test, y_train, y_test, 
                           model, feature_names, class_names):
    """Perform explainability analysis using LIME and SHAP"""
    print("\n" + "="*80)
    print("A5: EXPLAINABILITY WITH LIME AND SHAP")
    print("="*80)
    
    # Select a few test samples for explanation (ensure indices exist)
    max_idx = min(49, X_test.shape[0]-1)
    sample_indices = [i for i in (0, 5, 10) if i <= max_idx]
    if not sample_indices:
        sample_indices = [0]
    
    # ========================================================================
    # LIME ANALYSIS
    # ========================================================================
    print("\n--- LIME Analysis ---")
    
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )
    
    print("\nGenerating LIME explanations for sample instances...")
    
    fig, axes = plt.subplots(len(sample_indices), 1, figsize=(10, 4*len(sample_indices)))
    if len(sample_indices) == 1:
        axes = [axes]

    for idx, sample_idx in enumerate(sample_indices):
        exp = explainer_lime.explain_instance(
            X_test[sample_idx], 
            model.predict_proba,
            num_features=10
        )

        print(f"\nSample {sample_idx}:")
        print(f"  True class: {class_names[y_test[sample_idx]]}")
        print(f"  Predicted class: {class_names[model.predict([X_test[sample_idx]])[0]]}")
        print(f"  Top features:")
        for feat, weight in exp.as_list()[:10]:
            print(f"    {feat}: {weight:.4f}")

        # Create a horizontal bar chart from the LIME explanation list
        exp_list = exp.as_list()
        feat_names = [f[0] for f in exp_list]
        weights = [f[1] for f in exp_list]

        axes[idx].barh(range(len(weights)), weights, align='center')
        axes[idx].set_yticks(range(len(weights)))
        axes[idx].set_yticklabels(feat_names)
        axes[idx].invert_yaxis()
        axes[idx].set_xlabel('LIME feature weight')
        axes[idx].set_title(f'LIME Explanation - Sample {sample_idx}\n'
                           f'True: {class_names[y_test[sample_idx]]}, '
                           f'Pred: {class_names[model.predict([X_test[sample_idx]])[0]]}')
    
    plt.tight_layout()
    plt.savefig('lime_explanations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # SHAP ANALYSIS
    # ========================================================================
    print("\n--- SHAP Analysis ---")
    
    # Use TreeExplainer for tree-based models, or KernelExplainer for others
    try:
        explainer_shap = shap.TreeExplainer(model)
        shap_values = explainer_shap.shap_values(X_test[:50])  # Use subset for speed
    except:
        # Fallback to KernelExplainer
        explainer_shap = shap.KernelExplainer(model.predict_proba, X_train[:100])
        shap_values = explainer_shap.shap_values(X_test[:50])
    
    print("Generating SHAP visualizations...")
    
    # Summary plot
    plt.figure(figsize=(12, 8))
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[0], X_test[:50], feature_names=feature_names, 
                         show=False, max_display=20)
    else:
        shap.summary_plot(shap_values, X_test[:50], feature_names=feature_names, 
                         show=False, max_display=20)
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature importance plot
    plt.figure(figsize=(10, 8))
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[0], X_test[:50], feature_names=feature_names, 
                         plot_type='bar', show=False, max_display=20)
    else:
        shap.summary_plot(shap_values, X_test[:50], feature_names=feature_names, 
                         plot_type='bar', show=False, max_display=20)
    plt.tight_layout()
    plt.savefig('shap_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Force plot for individual predictions
    # Create robust per-sample SHAP bar charts (top absolute contributors)
    for sample_idx in sample_indices[:3]:  # first up to 3 samples
        try:
            if isinstance(shap_values, list):
                vals = shap_values[0][sample_idx]
            else:
                vals = shap_values[sample_idx]

            vals_arr = np.array(vals)
            abs_idx = np.argsort(np.abs(vals_arr))[::-1][:20]
            top_feats = [feature_names[i] if i < len(feature_names) else f'F{i}' for i in abs_idx]
            top_vals = vals_arr[abs_idx]

            plt.figure(figsize=(10, 6))
            plt.barh(range(len(top_vals)), top_vals[::-1], align='center')
            plt.yticks(range(len(top_vals)), [top_feats[i] for i in range(len(top_feats)-1, -1, -1)])
            plt.xlabel('SHAP value')
            plt.title(f'SHAP local explanation (top 20) - sample {sample_idx}')
            plt.tight_layout()
            plt.savefig(f'shap_sample_{sample_idx}.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print('SHAP per-sample plot failed for', sample_idx, e)
    
    print("\nSHAP visualizations saved!")
    
    # ========================================================================
    # COMPARISON OF LIME AND SHAP
    # ========================================================================
    print("\n" + "-"*80)
    print("COMPARISON: LIME vs SHAP")
    print("-"*80)
    
    print("""
LIME (Local Interpretable Model-agnostic Explanations):
  Strengths:
    - Model-agnostic: Works with any classifier
    - Fast for individual predictions
    - Intuitive local explanations
    - Easy to understand feature contributions for specific instances
  
  Limitations:
    - Only local explanations (instance-level)
    - May be unstable with different perturbations
    - Doesn't provide global feature importance directly
    - Sampling-based approach can be inconsistent
  
  Best Use Cases:
    - Explaining individual predictions to end users
    - Model debugging on specific cases
    - When you need model-agnostic explanations
    - Quick interpretability for black-box models

SHAP (SHapley Additive exPlanations):
  Strengths:
    - Theoretically grounded in game theory
    - Provides both local AND global explanations
    - Consistent and accurate feature attributions
    - Great visualizations (summary, force, dependence plots)
    - TreeExplainer is very fast for tree-based models
  
  Limitations:
    - Can be computationally expensive (KernelExplainer)
    - More complex to understand conceptually
    - Requires more computational resources
  
  Best Use Cases:
    - When you need rigorous, consistent explanations
    - Global feature importance analysis
    - Tree-based model interpretation (fast with TreeExplainer)
    - Academic or regulatory settings requiring theoretical soundness
    - Understanding overall model behavior

RECOMMENDATION FOR THIS DATASET:
  - Use SHAP for overall analysis (global feature importance)
  - Use LIME for explaining specific predictions to stakeholders
  - SHAP is preferred if using Random Forest/Gradient Boosting (fast TreeExplainer)
    """)
    
    return explainer_lime, explainer_shap

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("="*80)
    print("ELEPHANT CALL CLASSIFICATION ANALYSIS")
    print("Feature Selection, Dimensionality Reduction, and Explainability")
    print("="*80)
    
    # Load data
    X, y, feature_names, label_encoder = load_and_preprocess_data()
    
    # A1: Correlation Analysis
    corr_matrix = correlation_analysis(X, feature_names)
    
    # A2: PCA with 99% variance
    results_pca99, pca99, scaler99, X_train_pca99, X_test_pca99, y_train, y_test = \
        pca_analysis_99(X, y, feature_names)
    
    # A3: PCA with 95% variance
    results_pca95, pca95, scaler95 = pca_analysis_95(X, y, feature_names)
    
    # A4: Sequential Feature Selection
    results_sfs, sfs, selected_features = sequential_feature_selection(X, y, feature_names)
    
    # Compare all methods
    comparison_df = compare_results(results_pca99, results_pca95, results_sfs)
    
    # A5: Explainability - use PCA99 with Random Forest (best performing)
    print("\n\nPerforming explainability analysis on PCA (99%) with Random Forest...")
    best_model = results_pca99['Random Forest']['model']
    
    explainability_analysis(
        X_train_pca99, X_test_pca99, y_train, y_test,
        best_model, 
        [f'PC{i+1}' for i in range(X_train_pca99.shape[1])],
        label_encoder.classes_
    )
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - correlation_heatmap_full.png")
    print("  - correlation_heatmap_top30.png")
    print("  - pca_variance_99.png")
    print("  - pca_variance_95.png")
    print("  - method_comparison.png")
    print("  - lime_explanations.png")
    print("  - shap_summary.png")
    print("  - shap_importance.png")
    print("  - shap_force_sample_*.png")
    
    # Save comparison results
    comparison_df.to_csv('model_comparison_results.csv', index=False)
    print("  - model_comparison_results.csv")

if __name__ == "__main__":
    main()