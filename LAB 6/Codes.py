import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

file_path = r"D:\\OneDrive - Amrita vishwa vidyapeetham\\SEM 5\\23CSE301 Machine Learning\\LAB\\Machine-Learning\\LAB 3\\20231225_dfall_obs_data_and_spectral_features_revision1_n469.csv"
df = pd.read_csv(file_path)
df_clean = df.dropna(subset=['Call_Type', 'Quality'])
df_clean = df_clean[df_clean['Distance'] != 'NA'].copy()
df_clean['Distance'] = pd.to_numeric(df_clean['Distance'], errors='coerce')
df_clean = df_clean.dropna(subset=['Distance'])

print("-------------------------------------------------------------------------------------------------------------------------------")
# A1: Equal Width Binning and Entropy Calculation
def equal_width_binning(series, bins=4):
    try:
        return pd.cut(series, bins=bins, labels=False, include_lowest=True, duplicates='drop')          # Equal Width Binning
    except ValueError:                                                                                              
        return pd.qcut(series.rank(method='first'), q=bins, labels=False)                               # Equal Frequency Binning
def entropy(series):
    series_clean = pd.Series(series).dropna()                       # Remove NaN values
    if len(series_clean) == 0:                                      # If series is empty, return 0
        return 0                                                        
    counts = series_clean.value_counts()                            # Count occurrences of each value
    probabilities = counts / len(series_clean)                      # Calculate probabilities
    entropy_val = 0                                                 # Initialize entropy value
    for p in probabilities:                                         # Remove zero probabilities to avoid log2(0)
        if p > 0:
            entropy_val -= p * np.log2(p)                           # Calculate entropy
    return entropy_val                                              
df_clean['Distance_binned'] = equal_width_binning(df_clean['Distance'], bins=4)         # Binning Distance
df_clean['Quality_binned'] = equal_width_binning(df_clean['Quality'], bins=4)           # Binning Quality
entropy_distance = entropy(df_clean['Distance_binned'])                                 # Entropy for Distance
entropy_quality = entropy(df_clean['Quality_binned'])                                   # Entropy for Quality
print(f"A1 Result: ")
print("Entropy: Distance_binned={entropy_distance:.4f}, Quality_binned={entropy_quality:.4f}")

print("-------------------------------------------------------------------------------------------------------------------------------")
# A2: Gini Index Calculation
def gini_index(series):
    series_clean = pd.Series(series).dropna()                   # Remove NaN values
    if len(series_clean) == 0:                                  # If series is empty, return 0
        return 0    
    counts = series_clean.value_counts()                        # Count occurrences of each value
    probabilities = counts / len(series_clean)                  # Calculate probabilities
    gini_val = 1 - sum(p ** 2 for p in probabilities)           # Calculate Gini index
    return gini_val
gini_distance = gini_index(df_clean['Distance_binned'])         # Binning Distance
gini_quality = gini_index(df_clean['Quality_binned'])           # Binning Quality
print(f"A2 Result: ")
print(f"Gini Index: Distance_binned={gini_distance:.4f}, Quality_binned={gini_quality:.4f}")

print("-------------------------------------------------------------------------------------------------------------------------------")
# A3: Information Gain and Root Node Selection
def information_gain(df, feature, target):
    df_calc = df[[feature, target]].dropna()                            # Select relevant columns and drop NaN values#  
    if len(df_calc) == 0:                                               # If DataFrame is empty, return 0
        return 0
    total_entropy = entropy(df_calc[target])                            # Calculate total entropy for the target variable
    feature_values = df_calc[feature].unique()                          # Get unique values of the feature
    weighted_entropy = 0                                                # Initialize weighted entropy
    total_samples = len(df_calc)                                        # Total number of samples in the DataFrame
    for value in feature_values:
        subset = df_calc[df_calc[feature] == value]                     # Subset DataFrame based on feature value
        if len(subset) > 0:
            weight = len(subset) / total_samples                        # Calculate weight of the subset
            subset_entropy = entropy(subset[target])                    # Calculate entropy for the subset
            weighted_entropy += weight * subset_entropy                 # Add weighted entropy for the subset
    info_gain = total_entropy - weighted_entropy                        # Calculate information gain
    return info_gain                                                    
le_target = LabelEncoder()                                                                  # Label encoding for target variable
df_clean['Call_Type_encoded'] = le_target.fit_transform(df_clean['Call_Type'])              # Encode target variable
features_to_test = ['Distance_binned', 'Quality_binned']                                    # Features to test for information gain
information_gains = {}                                                                      # Initialize dictionary to store information gains
for feature in features_to_test:                                                        
    ig = information_gain(df_clean, feature, 'Call_Type_encoded')               # Calculate information gain for each feature
    information_gains[feature] = ig                                             # Store information gain in dictionary
best_root_feature = max(information_gains, key=information_gains.get)           # Select feature with maximum information gain
print(f"A3 Result: ")
for feature, gain in information_gains.items():
    print(f"Information Gain: {feature}={gain:.4f}")
print(f"Best root feature: {best_root_feature}")

print("--------------------------------------------------------------------------------------------------------------------------------")
# A4: Flexible Binning Function with Overloading
def flexible_binning(series, bins=4, strategy='uniform', bin_type='equal_width'):
    series_clean = series.dropna()                                                          # Remove NaN values
    if len(series_clean) == 0:                                                              # If series is empty, return empty array    
        return np.array([])     
    try:
        if bin_type == 'equal_frequency':                                              
            return pd.qcut(series_clean, q=bins, labels=False, duplicates='drop')               # Equal Frequency Binning
        else:
            enc = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=strategy)            # Equal Width Binning
            result = enc.fit_transform(series_clean.values.reshape(-1, 1))                      # Fit and transform the series
            return result.astype(int).flatten()                                                 # Return the result as a 1D array
    except Exception:
        return pd.cut(series_clean, bins=bins, labels=False, duplicates='drop')                                             # Fallback to pd.cut if KBinsDiscretizer fails
df_clean['CallerAge_binned'] = flexible_binning(df_clean['CallerAge'].fillna(df_clean['CallerAge'].median()), bins=3)       # Binning CallerAge
print(f"A4 Result: ")
print(f"CallerAge_binned: {df_clean['CallerAge_binned'].unique()}")

# print("------------------------------------------------------------------------------------------------------------------------------")
# A5: Build Custom Decision Tree Module
class CustomDecisionTree:
    def __init__(self, max_depth=5, min_samples_split=5, min_samples_leaf=2):           
        self.max_depth = max_depth                                                          # Maximum depth of the tree
        self.min_samples_split = min_samples_split                                          # Minimum samples required to split an internal node
        self.min_samples_leaf = min_samples_leaf                                            # Minimum samples required to be at a leaf node
        self.tree = {}                                                                      # Initialize the tree structure 
        self.feature_names = []                                                             # List to store feature names
        self.classes = []                                                                   # List to store unique classes in the target variable
    def fit(self, X, y):
        self.feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]      # Get feature names
        self.classes = np.unique(y)                                         # Unique classes in the target variable
        if not isinstance(X, pd.DataFrame):                         
            X = pd.DataFrame(X, columns=self.feature_names)                 # Convert X to DataFrame if not already
        if not isinstance(y, pd.Series):                                    
            y = pd.Series(y)                                                # Convert y to Series if not already
        self.tree = self._build_tree(X, y, depth=0)                         # Build the decision tree recursively
        return self                                                         # Return the tree structure
    def _build_tree(self, X, y, depth):
        if (depth >= self.max_depth or len(X) < self.min_samples_split or len(y.unique()) == 1 or len(X) < 2 * self.min_samples_leaf):
            return {'class': y.mode().iloc[0] if len(y) > 0 else self.classes[0], 'samples': len(y), 'distribution': y.value_counts().to_dict()}    # Leaf node condition
        best_feature, best_gain = self._find_best_split(X, y)                               # Find the best feature to split on
        if best_feature is None or best_gain <= 0:
            return {'class': y.mode().iloc[0] if len(y) > 0 else self.classes[0], 'samples': len(y), 'distribution': y.value_counts().to_dict()}    # If no valid split found, return leaf node
        tree_node = {'feature': best_feature, 'gain': best_gain, 'samples': len(y), 'distribution': y.value_counts().to_dict(), 'children': {}}       # Create a tree node with the best feature and its gain
        feature_values = X[best_feature].unique()                                                           # Get unique values of the best feature
        for value in feature_values:    
            mask = X[best_feature] == value                                                                 # Create a mask for the current feature value
            if mask.sum() >= self.min_samples_leaf:                                                         # Ensure enough samples for the child node
                child_X = X[mask]                                                                           # Subset X based on the mask
                child_y = y[mask]                                                                           # Subset y based on the mask
                tree_node['children'][value] = self._build_tree(child_X, child_y, depth + 1)                # Recursively build the child node
        if not tree_node['children']:   
            return {'class': y.mode().iloc[0] if len(y) > 0 else self.classes[0], 'samples': len(y), 'distribution': y.value_counts().to_dict()}    # If no children were created, return a leaf node
        return tree_node                                            # Return the tree node with its children
    def _find_best_split(self, X, y):   
        best_gain = -1                                                  # Initialize best gain
        best_feature = None                                             # Initialize best feature                                   
        for feature in X.columns:                                       # Iterate through each feature
            if X[feature].nunique() <= 1:                               # Skip features with only one unique value
                continue
            df_temp = pd.concat([X[[feature]], y], axis=1)                                          # Create a temporary DataFrame with the feature and target variable
            gain = information_gain(df_temp, feature, y.name if y.name else 'target')               # Calculate information gain for the feature
            if gain > best_gain:                                        
                best_gain = gain                                # Update best gain if current gain is better
                best_feature = feature                          # Update best feature if current gain is better
        return best_feature, best_gain                          # Return the best feature and its gain
    def predict(self, X):
        if not isinstance(X, pd.DataFrame):                                 # Convert X to DataFrame if not already
            X = pd.DataFrame(X, columns=self.feature_names)                 # Ensure X has the same feature names as used during training
        predictions = []                                                # Initialize list to store predictions
        for idx, row in X.iterrows():                                   # Iterate through each row in X
            pred = self._predict_single(row, self.tree)                 # Predict the class for the current sample
            predictions.append(pred)                                    # Append the prediction to the list
        return np.array(predictions)                                    # Return predictions as a NumPy array
    def _predict_single(self, sample, node):                            # Recursive function to predict a single sample
        if 'class' in node:
            return node['class']                                        # If leaf node, return the class
        feature = node['feature']                                   
        if feature not in sample:                                       # If feature not in sample, return the most common class in the node
            return self.classes[0]                                      
        value = sample[feature]                                             # Get the value of the feature in the sample
        if value in node['children']:                                                       
            return self._predict_single(sample, node['children'][value])                # If value exists in children, recursively predict using the child node
        else:
            if 'distribution' in node and node['distribution']:
                return max(node['distribution'], key=node['distribution'].get)          # If value does not exist in children, return the most common class in the node
            return self.classes[0]                                                      # If no children exist, return the first class in the classes list
features_for_custom_tree = ['Distance_binned', 'Quality_binned']
X_custom = df_clean[features_for_custom_tree].dropna()
y_custom = df_clean.loc[X_custom.index, 'Call_Type_encoded']
custom_dt = CustomDecisionTree(max_depth=4, min_samples_split=3)
custom_dt.fit(X_custom, y_custom)

# print("------------------------------------------------------------------------------------------------------------------------------")
# A6: Decision Tree Visualization
X_viz = df_clean[['Distance_binned', 'Quality_binned']].dropna()                      # Select features for visualization
y_viz = df_clean.loc[X_viz.index, 'Call_Type_encoded']                                # Select target variable for visualization
dt_viz = DecisionTreeClassifier(max_depth=4, min_samples_split=5, min_samples_leaf=2, random_state=42)   # Initialize DecisionTreeClassifier
dt_viz.fit(X_viz, y_viz)                                                             # Fit the classifier
plt.figure(figsize=(20, 12))                                                         # Set figure size
class_names = [f'{cls}' for cls in le_target.classes_]                               # Get class names
plot_tree(dt_viz, 
          feature_names=['Distance_binned', 'Quality_binned'],
          class_names=class_names,
          filled=True, 
          rounded=True, 
          fontsize=12,
          max_depth=3)                                                               # Plot the decision tree
plt.title("Decision Tree for Elephant Call Classification", fontsize=16, pad=20)     # Set plot title
plt.tight_layout()                                                                   # Adjust layout
plt.show()                                                                           # Show plot

# print("------------------------------------------------------------------------------------------------------------------------------")
# A7: Decision Boundary Visualization
X_boundary = X_viz.copy()                                                            # Use same features for boundary visualization
y_boundary = y_viz.copy()                                                            # Use same target for boundary visualization
X_train, X_test, y_train, y_test = train_test_split(
    X_boundary, y_boundary, test_size=0.3, random_state=42, stratify=y_boundary      # Split data for training and testing
)
dt_boundary = DecisionTreeClassifier(max_depth=4, min_samples_split=5, min_samples_leaf=2, random_state=42)   # Initialize DecisionTreeClassifier
dt_boundary.fit(X_train, y_train)                                                    # Fit the classifier
y_pred = dt_boundary.predict(X_test)                                                 # Predict on test data
accuracy = accuracy_score(y_test, y_pred)                                            # Calculate accuracy
plt.figure(figsize=(15, 10))                                                         # Set figure size
h = 0.02
x_min, x_max = X_boundary.iloc[:, 0].min() - 0.5, X_boundary.iloc[:, 0].max() + 0.5 # Set mesh grid range for X
y_min, y_max = X_boundary.iloc[:, 1].min() - 0.5, X_boundary.iloc[:, 1].max() + 0.5 # Set mesh grid range for Y
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))        # Create mesh grid
try:
    Z = dt_boundary.predict(np.c_[xx.ravel(), yy.ravel()])                          # Predict on mesh grid
    Z = Z.reshape(xx.shape)                                                         # Reshape predictions
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.Set3)                            # Plot decision boundary
    colors = plt.cm.Set3(np.linspace(0, 1, len(np.unique(y_boundary))))             # Get colors for classes
    for i, class_label in enumerate(np.unique(y_boundary)):
        mask = y_boundary == class_label
        plt.scatter(X_boundary.iloc[mask, 0], X_boundary.iloc[mask, 1], 
                   c=[colors[i]], label=f'{le_target.classes_[class_label]}',
                   edgecolors='black', s=50, alpha=0.8)                             # Plot data points
    plt.xlabel('Distance (binned)', fontsize=14)
    plt.ylabel('Quality (binned)', fontsize=14)
    plt.title(f'Decision Boundary for Elephant Call Classification\n(Accuracy: {accuracy:.3f})', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')                          # Add legend
    plt.grid(True, alpha=0.3)                                                       # Add grid
    plt.tight_layout()                                                              # Adjust layout 
    plt.show()                                                                      # Show plot
except Exception:
    plt.scatter(X_boundary.iloc[:, 0], X_boundary.iloc[:, 1], c=y_boundary, cmap=plt.cm.Set3)       # Plot data points
    plt.xlabel('Distance (binned)')                                                                 
    plt.ylabel('Quality (binned)')
    plt.title('Elephant Call Data Distribution')
    plt.colorbar()
    plt.show()
print(f"A7 - Decision boundary visualized, accuracy={accuracy:.4f}")
