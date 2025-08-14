import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# === Load the attached dataset ===
file_path = r"D:\\OneDrive - Amrita vishwa vidyapeetham\\SEM 5\\23CSE301 Machine Learning\\LAB\\Machine-Learning\\LAB 3\\20231225_dfall_obs_data_and_spectral_features_revision1_n469.csv"
df = pd.read_csv(file_path)

# === A1: Equal Width Binning & Entropy Calculation ===
def equal_width_binning(series, bins=4):
    return pd.cut(series, bins=bins, labels=False, duplicates='drop')

def frequency_binning(series, bins=4):
    return pd.qcut(series, q=bins, labels=False, duplicates='drop')

def flexible_binning(series, bins=4, binning_type='equal_width'):
    if binning_type == 'equal_width':
        return equal_width_binning(series, bins)
    elif binning_type == 'frequency':
        return frequency_binning(series, bins)
    else:
        raise ValueError("binning_type must be 'equal_width' or 'frequency'")

def entropy(series):
    if len(series) == 0:
        return 0
    counts = series.value_counts()
    probabilities = counts / len(series)
    return -sum(p * np.log2(p) for p in probabilities if p > 0)

# === A2: Gini Index Calculation ===
def gini_index(series):
    if len(series) == 0:
        return 0
    counts = series.value_counts()
    probabilities = counts / len(series)
    return 1 - sum(p ** 2 for p in probabilities)

# === A3: Information Gain for Root Node Selection ===
def information_gain(df, feature, target):
    total_entropy = entropy(df[target])
    values = df[feature].unique()
    weighted_entropy = 0
    for v in values:
        subset = df[df[feature] == v]
        if len(subset) > 0:
            weight = len(subset) / len(df)
            weighted_entropy += weight * entropy(subset[target])
    return total_entropy - weighted_entropy

# === Data Preprocessing ===
available_columns = df.columns.tolist()
# Select two features and one target for demonstration (customize as needed)
feature_columns = available_columns[1:3]  # Use columns 1 and 2 as features
target_column = available_columns[-1]     # Use last column as target
working_df = df.copy()
for col in feature_columns + [target_column]:
    if col in working_df.columns:
        if working_df[col].dtype == 'object':
            working_df[col] = working_df[col].fillna('Unknown')
        else:
            working_df[col] = working_df[col].fillna(working_df[col].median())

# === A1: Equal Width Binning & Entropy Calculation ===
binned_features = {}
for feature in feature_columns:
    if feature in working_df.columns:
        if working_df[feature].dtype in ['int64', 'float64']:
            binned_name = f"{feature}_binned"
            working_df[binned_name] = flexible_binning(working_df[feature], bins=4, binning_type='equal_width')
            binned_features[feature] = binned_name
            entropy_val = entropy(working_df[binned_name])
            print(f"A1: Entropy of {binned_name}: {entropy_val:.4f}")
        else:
            entropy_val = entropy(working_df[feature])
            print(f"A1: Entropy of {feature}: {entropy_val:.4f}")
if target_column in working_df.columns:
    # Bin target if it's continuous
    if working_df[target_column].dtype in ['int64', 'float64']:
        target_binned = flexible_binning(working_df[target_column], bins=4, binning_type='equal_width')
        working_df[target_column + '_binned'] = target_binned
        target_entropy = entropy(target_binned)
        print(f"A1: Entropy of target ({target_column}_binned): {target_entropy:.4f}")
        target_column_used = target_column + '_binned'
    else:
        target_entropy = entropy(working_df[target_column])
        print(f"A1: Entropy of target ({target_column}): {target_entropy:.4f}")
        target_column_used = target_column

# === A2: Gini Index Calculation ===
for feature in feature_columns:
    if feature in working_df.columns:
        feature_to_use = binned_features[feature] if feature in binned_features else feature
        gini_val = gini_index(working_df[feature_to_use])
        print(f"A2: Gini Index of {feature_to_use}: {gini_val:.4f}")
if target_column in working_df.columns:
    target_gini = gini_index(working_df[target_column_used])
    print(f"A2: Gini Index of target ({target_column_used}): {target_gini:.4f}")

# === A3: Information Gain and Root Node Selection ===
gains = {}
for feature in feature_columns:
    if feature in working_df.columns:
        if feature in binned_features:
            feature_to_use = binned_features[feature]
        else:
            if working_df[feature].dtype == 'object':
                le = LabelEncoder()
                working_df[f"{feature}_encoded"] = le.fit_transform(working_df[feature])
                feature_to_use = f"{feature}_encoded"
            else:
                feature_to_use = feature
        try:
            gain = information_gain(working_df, feature_to_use, target_column_used)
            gains[feature_to_use] = gain
            print(f"A3: Information Gain for {feature_to_use}: {gain:.4f}")
        except Exception as e:
            print(f"A3: Could not calculate information gain for {feature}: {e}")
if gains:
    root_feature = max(gains, key=gains.get)
    print(f"A3: Root node feature: {root_feature} (Information Gain: {gains[root_feature]:.4f})")

# === A4: Decision Tree Construction & Visualization ===
X_features = []
feature_names = []
for feature in feature_columns:
    if feature in working_df.columns:
        if feature in binned_features:
            X_features.append(working_df[binned_features[feature]].values)
            feature_names.append(binned_features[feature])
        else:
            if working_df[feature].dtype == 'object':
                le = LabelEncoder()
                encoded_feature = le.fit_transform(working_df[feature])
                X_features.append(encoded_feature)
                feature_names.append(f"{feature}_encoded")
            else:
                X_features.append(working_df[feature].values)
                feature_names.append(feature)
if X_features and target_column in working_df.columns:
    X = np.column_stack(X_features)
    # Use binned target column for classification
    if working_df[target_column_used].dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(working_df[target_column_used])
        class_names = le_target.classes_
    else:
        y = working_df[target_column_used].values
        class_names = [str(i) for i in sorted(np.unique(y))]
    valid_indices = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[valid_indices]
    y = y[valid_indices]
    clf = DecisionTreeClassifier(max_depth=4, min_samples_split=5, random_state=42)
    clf.fit(X, y)
    plt.figure(figsize=(15, 10))
    plot_tree(clf, feature_names=feature_names, class_names=[str(c) for c in class_names], filled=True, rounded=True, fontsize=10)
    plt.title("A4: Decision Tree Visualization")
    plt.tight_layout()
    plt.show()
    # === A5: Feature Importance ===
    feature_importance = clf.feature_importances_
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance}).sort_values('importance', ascending=False)
    print("A5: Feature Importance:")
    print(importance_df)
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Importance')
    plt.title('A5: Feature Importance in Decision Tree')
    plt.tight_layout()
    plt.show()
    # === A6: Model Performance ===
    if len(X) > 10:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        clf_eval = DecisionTreeClassifier(max_depth=4, min_samples_split=5, random_state=42)
        clf_eval.fit(X_train, y_train)
        train_accuracy = accuracy_score(y_train, clf_eval.predict(X_train))
        test_accuracy = accuracy_score(y_test, clf_eval.predict(X_test))
        print(f"A6: Training Accuracy: {train_accuracy:.4f}")
        print(f"A6: Testing Accuracy: {test_accuracy:.4f}")
    # === A7: Decision Boundary Visualization (for first 2 features) ===
    if X.shape[1] >= 2:
        # Use only first two features for boundary plot
        X2 = X[:, :2]
        # If both features are binned, use integer meshgrid
        bins_x = int(np.nanmax(X2[:, 0]) + 1)
        bins_y = int(np.nanmax(X2[:, 1]) + 1)
        xx, yy = np.meshgrid(np.arange(0, bins_x), np.arange(0, bins_y))
        grid = np.c_[xx.ravel(), yy.ravel()]
        # If more than 2 features, pad with zeros
        if X.shape[1] > 2:
            pad = np.zeros((grid.shape[0], X.shape[1] - 2))
            grid = np.hstack([grid, pad])
        Z = clf.predict(grid)
        Z = Z.reshape(xx.shape)
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(len(np.unique(y))+1)-0.5, cmap=plt.cm.Set1)
        scatter = plt.scatter(X2[:, 0], X2[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.title('A7: Decision Boundary (First 2 Features, Binned)')
        plt.colorbar(scatter, ticks=np.unique(y))
        plt.tight_layout()
        plt.show()