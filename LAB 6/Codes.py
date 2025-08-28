import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

file_path = r"D:\\OneDrive - Amrita vishwa vidyapeetham\\SEM 5\\23CSE301 Machine Learning\\LAB\\Machine-Learning\\LAB 3\\20231225_dfall_obs_data_and_spectral_features_revision1_n469.csv"
df = pd.read_csv(file_path)

print("-------------------------------------------------------------------------------------------------------------------------------")
# A1: Equal Width Binning and Entropy Calculation
def equal_width_binning(series, bins=4):
    return pd.cut(series, bins=bins, labels=False, include_lowest=True, duplicates='drop')  # Equal Width Binning

# Function to calculate entropy
def entropy(series):
    series_clean = pd.Series(series).dropna()       # Remove NaN values
    if len(series_clean) == 0:                      # If series is empty, return 0
        return 0
    counts = series_clean.value_counts()            # Count occurrences of each value
    probabilities = counts / len(series_clean)      # Calculate probabilities
    entropy_val = 0                                 # Initialize entropy value
    for p in probabilities:                         # Remove zero probabilities to avoid log2(0)
        if p > 0:
            entropy_val -= p * np.log2(p)           # Calculate entropy
    return entropy_val

df['CallerAge_binned'] = equal_width_binning(df['CallerAge'], bins=4)       # Apply binning to 'CallerAge' column
callerage_distance = entropy(df['CallerAge_binned'])                        # Calculate entropy for binned 'CallerAge'
print(f"A1 Result: ")
print(f"Entropy: CallerAge_binned = {callerage_distance:.4f}")

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
gini_distance = gini_index(df['CallerAge_binned'])         # Binning Distance
print(f"A2 Result: ")
print(f"Gini Index: CallerAge_binned = {gini_distance:.4f}")

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
features_to_test = ['F1', 'F2', 'F3', 'F4']                                    # Features to test for information gain
information_gains = {}                                                         # Initialize dictionary to store information gains
for feature in features_to_test:                                                        
    ig = information_gain(df, feature, 'CallerAge')                             # Calculate information gain for each feature
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
        return pd.cut(series_clean, bins=bins, labels=False, duplicates='drop')                 # Fallback to pd.cut if KBinsDiscretizer fails
df['F1_cat'] = flexible_binning(df['F1'], bins=4, strategy='uniform')           #Binning F1
df['F2_cat'] = flexible_binning(df['F2'], bins=4, strategy='uniform')           #Binning F2
df['F3_cat'] = flexible_binning(df['F3'], bins=4, strategy='uniform')           #Binning F3
df['F4_cat'] = flexible_binning(df['F4'], bins=4, strategy='uniform')           #Binning F4
print(f"A4 Result: ")
print(f"F1_binned: {df['F1_cat'].unique()}")
print(f"F2_binned: {df['F2_cat'].unique()}")
print(f"F3_binned: {df['F3_cat'].unique()}")
print(f"F4_binned: {df['F4_cat'].unique()}")

print("------------------------------------------------------------------------------------------------------------------------------")

# A5: Build Custom Decision Tree Module
binned_features = ['F1_cat', 'F2_cat', 'F3_cat', 'F4_cat']
# Compute information gain for each feature with respect to 'CallerSex'
gains = {}
for feature in binned_features:
    gains[feature] = information_gain(df, feature, 'CallerSex')
# Identify the root node feature based on the highest information gain
root_feature = max(gains, key=gains.get)
print("A5 Answer: ")
print("Root node feature based on Information Gain:", root_feature)

print("------------------------------------------------------------------------------------------------------------------------------")

# A6: Decision Tree Visualization
X = df[['F1_cat', 'F2_cat', 'F3_cat', 'F4_cat']]  # Binned features
y = df['CallerSex']                             # Target variable
# Train a Decision Tree Classifier
clf = DecisionTreeClassifier(criterion='entropy')  # Using entropy to match information gain logic
clf.fit(X, y)
# Visualize the decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True, rounded=True, fontsize=12)
plt.title("Decision Tree for Predicting CallerSex", fontsize=15)
plt.show()

# print("------------------------------------------------------------------------------------------------------------------------------")
# A7: Decision Boundary Visualization
df['CallerSex'] = df['CallerSex'].map({'F': 0, 'M': 1})
df_cleaned = df[np.isfinite(df[['F1', 'F2']]).all(axis=1)]

X = df_cleaned[['F1', 'F2']] 
y = df_cleaned['CallerSex']  # Target variable (CallerSex)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train the Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Set up the mesh grid for plotting the decision regions
x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Predict the class for each point in the mesh grid
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3)  # Decision boundary regions
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)  # Scatter plot of points
plt.xlabel('F1')  # X-axis label (F1)
plt.ylabel('F2')  # Y-axis label (F2)
plt.title('Decision Boundary of Decision Tree for CallerSex')
plt.show()

# print("------------------------------------------------------------------------------------------------------------------------------")