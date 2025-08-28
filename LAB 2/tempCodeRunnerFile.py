import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_excel(xls, sheet_name="marketing_campaign")

# Subset the first 20 vectors
df_subset = df.iloc[:20, :].copy()

# Identify binary columns (values 0 and 1)
binary_cols = [col for col in df.columns if set(df_subset[col].dropna().unique()).issubset({0, 1})]

# Identify numeric columns (all numeric columns in the subset)
numeric_cols_20 = [col for col in df.columns if col in df_subset.columns and pd.api.types.is_numeric_dtype(df_subset[col])]

# Convert numeric columns to numeric and fill NaNs with the median of the column
for col in numeric_cols_20:
    df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce')
    df_subset[col] = df_subset[col].fillna(df_subset[col].median())

# Initialize similarity matrices
JC_matrix = np.zeros((20, 20))
SMC_matrix = np.zeros((20, 20))
COS_matrix = np.zeros((20, 20))

# Compute the similarities
for i in range(20):
    for j in range(20):
        # Binary column vectors for i-th and j-th observations
        v1 = df_subset.iloc[i][binary_cols].values if binary_cols else np.array([])
        v2 = df_subset.iloc[j][binary_cols].values if binary_cols else np.array([])
        
        if len(v1) > 0 and len(v2) > 0:  # Only compute for binary columns if they exist
            f11 = np.sum((v1 == 1) & (v2 == 1))  # Matches where both are 1
            f00 = np.sum((v1 == 0) & (v2 == 0))  # Matches where both are 0
            f10 = np.sum((v1 == 1) & (v2 == 0))  # Matches where first is 1 and second is 0
            f01 = np.sum((v1 == 0) & (v2 == 1))  # Matches where first is 0 and second is 1
            
            # Jaccard Coefficient (JC): f11 / (f11 + f10 + f01)
            denominator = f11 + f10 + f01
            JC_matrix[i, j] = f11 / denominator if denominator != 0 else 0

            # Simple Matching Coefficient (SMC): (f11 + f00) / (f11 + f10 + f01 + f00)
            total_elements = f11 + f10 + f01 + f00
            SMC_matrix[i, j] = (f11 + f00) / total_elements if total_elements != 0 else 0
        else:
            JC_matrix[i, j] = np.nan  # No binary values for the pair
            SMC_matrix[i, j] = np.nan  # No binary values for the pair
        
        # Numeric column vectors for i-th and j-th observations
        v1_num = df_subset.iloc[i][numeric_cols_20].values if numeric_cols_20 else np.array([])
        v2_num = df_subset.iloc[j][numeric_cols_20].values if numeric_cols_20 else np.array([])

        if len(v1_num) > 0 and len(v2_num) > 0:  # Only compute if numeric values exist
            # Compute Cosine Similarity
            COS_matrix[i, j] = cosine_similarity([v1_num], [v2_num])[0][0]
        else:
            COS_matrix[i, j] = np.nan  # No numeric values for the pair

# Plotting the heatmaps
plt.figure(figsize=(24, 7))

# Plot Jaccard Coefficient heatmap
plt.subplot(1, 3, 1)
sns.heatmap(JC_matrix, annot=False, cmap='YlGnBu', vmin=0, vmax=1)
plt.title('Jaccard Coefficient (JC)')

# Plot Simple Matching Coefficient heatmap
plt.subplot(1, 3, 2)
sns.heatmap(SMC_matrix, annot=False, cmap='YlOrRd', vmin=0, vmax=1)
plt.title('Simple Matching Coefficient (SMC)')

# Plot Cosine Similarity heatmap
plt.subplot(1, 3, 3)
sns.heatmap(COS_matrix, annot=False, cmap='coolwarm', vmin=0, vmax=1)
plt.title('Cosine Similarity (COS)')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

print("A7 Results: JC, SMC, and COS similarities visualized for first 20 vectors.")