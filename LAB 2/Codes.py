# Import necessary libraries
import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression

# Defining the file path to the Excel workbook containing the datasets
file_path = r"D:\\OneDrive - Amrita vishwa vidyapeetham\\SEM 5\\23CSE301 Machine Learning\\LAB\\Machine-Learning\\LAB 2\\Lab Session Data.xlsx"  
xls = pd.ExcelFile(file_path)

# A1: Matrix operations on purchase data to estimate product costs
df = pd.read_excel(xls, sheet_name="Purchase data")                           # Reading the "Purchase data" sheet into a DataFrame
purchase_matrix = df.iloc[:, 1:4].values                                      # Extracting purchase matrix (quantities of products purchased) -> Matrix A
purchase_amounts = df.iloc[:, 4].values                                       # Extracting purchase amounts (total cost per purchase) -> Matrix B
dimensionality = purchase_matrix.shape[1]                                     # Dimensionality of the vector space
num_vectors = purchase_matrix.shape[0]                                        # Number of vectors (purchases)
rank_A = np.linalg.matrix_rank(purchase_matrix)                               # Rank of the purchase matrix (A)
purchase_matrix_pinv = np.linalg.pinv(purchase_matrix)                        # Computing Pseudo-Inverse
product_costs = np.dot(purchase_matrix_pinv, purchase_amounts)                # Calculating product costs using pseudo-inverse
print("A1 Answers:")
print(f"    Dimensionality of the vector space: {dimensionality}")
print(f"    Number of Vectors: {num_vectors}")
print(f"    Rank of Matrix A: {rank_A}")
print(f"    Product Costs: {product_costs}")

print("----------------------------------------------------------------------------------------------------------")

# A2: Classifying customers based on purchase amounts
df = pd.read_excel(xls, sheet_name="Purchase data")
df["Customer Class"] = np.where(df.iloc[:, 4] > 200, "RICH", "POOR")            # Classifying customers based on purchase amounts
print("A2 Answers:")
print(df["Customer Class"])

print("----------------------------------------------------------------------------------------------------------")

#A3: Analyzing IRCTC stock price data
df = pd.read_excel(xls, sheet_name="IRCTC Stock Price")                       # Reading the "IRCTC Stock Price" sheet into a DataFrame
mean_price = statistics.mean(df["Price"])                                     # Calculating mean price
variance_price = statistics.variance(df["Price"])                             # Calculating variance of price
wednesday_mean = df[df["Day"] == "Wed"]["Price"].mean()                       # Calculating mean price for Wednesdays
april_mean = df[df["Month"] == "Apr"]["Price"].mean()                     # Calculating mean price for April
prob_loss = (df["Chg%"] < 0).mean()                                           # Calculating probability of loss
prob_profit_wed = df[(df["Day"] == "Wednesday") & (df["Chg%"] > 0)]["Chg%"].count() / df[df["Day"] == "Wednesday"]["Chg%"].count() # Calculating probability of profit on Wednesdays
print("A3 Answers:")
print(f"    Mean Price: {mean_price}")
print(f"    Variance Price: {variance_price}")
print(f"    Wednesday Mean Price: {wednesday_mean}")
print(f"    April Mean Price: {april_mean}")
print(f"    Probability of Loss: {prob_loss}")
print(f"    Probability of Profit on Wednesday: {prob_profit_wed}")
plt.figure(figsize=(10, 5))                             # Sets the figure size for the plot
sns.scatterplot(x=df["Day"], y=df["Chg%"])              # Scatter plot of Change % vs. Day of the Week
plt.xlabel("Day of the Week")                           
plt.ylabel("Change %")
plt.xticks(rotation=45)                                 # Rotates x-axis labels for better readability
plt.title("Change % vs. Day of the Week")
plt.tight_layout()                                      # prevents labels from overlapping
plt.show()                                              # Displays the scatter plot of Change % vs. Day of the Week

print("----------------------------------------------------------------------------------------------------------")

# A4: Data Exploration
df = pd.read_excel(xls, sheet_name="thyroid0387_UCI")               # Reading the "thyroid0387_UCI" sheet into a DataFrame
df.replace('?', np.nan, inplace=True)                               # Replaces '?' with NaN for proper handling of missing values
df = df.infer_objects()                                             # Ensures proper type conversion
missing_values = df.isnull().sum()                                  # Counts missing values in each column
nominal_cols = df.select_dtypes(include=['object']).columns.tolist()                # All categorical column -> Nominal columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()      # Numeric columns for analysis
df_encoded = pd.get_dummies(df, columns=nominal_cols, drop_first=True)          # One-hot encoding for nominal columns
print("A4 Results: ")
print("\nNumeric Variable Ranges:")
for col in numeric_cols:                                            
    print(f"{col}: min={df[col].min()}, max={df[col].max()}")                   # Displaying min and max for numeric variables
print("\nMissing Values per Attribute:\n", missing_values)
print("\nOutlier Detection (IQR method):")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)                                     # First quartile (25th percentile)
    Q3 = df[col].quantile(0.75)                                     # Third quartile (75th percentile)    
    IQR = Q3 - Q1                                                   # Interquartile range
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)][col]         # Identifying outliers using IQR method
    print(f"{col}: {len(outliers)} outliers")                                          
print("\nMean and Std for Numeric Variables:")
for col in numeric_cols:
    print(f"{col}: mean={df[col].mean()}, std={df[col].std()}")             # Displaying mean and standard deviation for numeric variables

print("----------------------------------------------------------------------------------------------------------")

# A5: Jaccard Coefficient and SMC (using only binary attributes)
df = pd.read_excel(xls, sheet_name="marketing_campaign")
binary_cols = [col for col in df.columns if set([df.iloc[0][col], df.iloc[1][col]]).issubset({0, 1})]       # Binary columns
v1 = df.iloc[0][binary_cols].values                     # Values of the first row for binary columns
v2 = df.iloc[1][binary_cols].values                     # Values of the second row for binary columns
f11 = np.sum((v1 == 1) & (v2 == 1))                     # Count of matches where both are 1
f00 = np.sum((v1 == 0) & (v2 == 0))                     # Count of matches where both are 0
f10 = np.sum((v1 == 1) & (v2 == 0))                     # Count of matches where first is 1 and second is 0
f01 = np.sum((v1 == 0) & (v2 == 1))                     # Count of matches where first is 0 and second is 1
denominator = (f01 + f10 + f11)                                                             # Denominator for Jaccard Coefficient calculation
JC = f11 / denominator if denominator != 0 else 0                                           # Handles the case where all are 0
SMC = (f11 + f00) / (f00 + f01 + f10 + f11) if (f00 + f01 + f10 + f11) != 0 else 0          # Handles the case where all are 0
print("A5 Results:")
print(f"Jaccard Coefficient: {JC}, SMC: {SMC}")
print(f"Number of binary attributes considered: {len(binary_cols)}")
if JC > SMC:                    # Indicates more shared positive matches
    print("JC is higher, indicating more shared positive matches. JC is appropriate when only positive matches matter.")
elif JC < SMC:                  # Indicates more shared matches including negatives
    print("SMC is higher, indicating both matches (0 and 1) are considered. SMC is appropriate when both matches are important.")
else:                           # Indicates both are equal
    print("JC and SMC are equal, indicating similar matching for this pair.")

print("----------------------------------------------------------------------------------------------------------")

# A6: Cosine Similarity Calculation
df = pd.read_excel(xls, sheet_name="thyroid0387_UCI")               # Reading the "thyroid0387_UCI" sheet into a DataFrame
df.replace('?', np.nan, inplace=True)                               # Replaces '?' with NaN for proper handling of missing values
df = df.infer_objects()                                             # Ensures proper type conversion
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()      # Numeric columns for analysis
df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.median()), axis=0)
vector1 = df.iloc[0][numeric_cols].values.reshape(1, -1)  # First row, only numeric columns
vector2 = df.iloc[1][numeric_cols].values.reshape(1, -1)  # Second row, only numeric columns
result = cosine_similarity(vector1, vector2)[0][0]
print("A6 Result:", result)

print("----------------------------------------------------------------------------------------------------------")

# A7: Heatmap of JC, SMC, and COS similarities for first 20 vectors

df = pd.read_excel(xls, sheet_name="marketing_campaign")
df_subset = df.iloc[:20, :].copy()                           # Subset the first 20 vectors
binary_cols = [col for col in df.columns if set(df_subset[col].dropna().unique()).issubset({0, 1})]      # Identify binary columns (values 0 and 1)
numeric_cols_20 = [col for col in df.columns if col in df_subset.columns and pd.api.types.is_numeric_dtype(df_subset[col])]     # Identify numeric columns (all numeric columns in the subset)
for col in numeric_cols_20:
    df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce')             # Convert to numeric, coercing errors to NaN
    df_subset[col] = df_subset[col].fillna(df_subset[col].median())             # Fill NaNs in numeric columns with the median
# Initialize similarity matrices
JC_matrix = np.zeros((20, 20))
SMC_matrix = np.zeros((20, 20))
COS_matrix = np.zeros((20, 20))
for i in range(20):
    for j in range(20):
        v1 = df_subset.iloc[i][binary_cols].values if binary_cols else np.array([])         # Binary column vectors for i-th observation
        v2 = df_subset.iloc[j][binary_cols].values if binary_cols else np.array([])         # Binary column vectors for j-th observation
        if len(v1) > 0 and len(v2) > 0:                     # Only compute for binary columns if they exist
            f11 = np.sum((v1 == 1) & (v2 == 1))             # Matches where both are 1
            f00 = np.sum((v1 == 0) & (v2 == 0))             # Matches where both are 0
            f10 = np.sum((v1 == 1) & (v2 == 0))             # Matches where first is 1 and second is 0
            f01 = np.sum((v1 == 0) & (v2 == 1))             # Matches where first is 0 and second is 1  
            # Jaccard Coefficient (JC): f11 / (f11 + f10 + f01)
            denominator = f11 + f10 + f01
            JC_matrix[i, j] = f11 / denominator if denominator != 0 else 0
            # Simple Matching Coefficient (SMC): (f11 + f00) / (f11 + f10 + f01 + f00)
            total_elements = f11 + f10 + f01 + f00
            SMC_matrix[i, j] = (f11 + f00) / total_elements if total_elements != 0 else 0
        else:
            JC_matrix[i, j] = np.nan        # No binary values for the pair
            SMC_matrix[i, j] = np.nan       # No binary values for the pair
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


print("----------------------------------------------------------------------------------------------------------")

# A8: Handling missing values and scaling
df = pd.read_excel(xls, sheet_name="thyroid0387_UCI")                       # Reading the "thyroid0387_UCI" sheet into a DataFrame
df.replace('?', np.nan, inplace=True)                                       # Replaces '?' with NaN for proper handling of missing values
df.infer_objects(copy=False)                                                # Ensures proper type conversion
for col in df.columns:                                                      
    if df[col].dtype in ['float64', 'int64']:                               # Numerical columns
        Q1 = df[col].quantile(0.25)                                         # First quartile (25th percentile)      
        Q3 = df[col].quantile(0.75)                                         # Third quartile (75th percentile)
        IQR = Q3 - Q1                                                                       # Interquartile range                   
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)][col]         # Outliers are those below Q1 - 1.5*IQR or above Q3 + 1.5*IQR
        if len(outliers) > 0:
            df[col] = df[col].fillna(df[col].median())                      # Use median for numerical columns with outliers
        else:
            df[col] = df[col].fillna(df[col].mean())                        # Use mean for numerical columns without outliers
    else:
        df[col] = df[col].fillna(df[col].mode()[0])                          # Categorical columns, fill NaN with mode
print("A8 Results:")
print(df)

print("----------------------------------------------------------------------------------------------------------")

# A9: Encoding categorical variables and scaling numerical variables
categorical_cols = df.select_dtypes(include=['object']).columns                     # Select categorical columns
for col in categorical_cols:                                                        # Encode categorical variables using Label Encoding
    df[col] = LabelEncoder().fit_transform(df[col])                                 # Convert categorical columns to numeric using Label Encoding
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns             # Select numerical columns for scaling
scaler = MinMaxScaler()                                                             # Initialize MinMaxScaler for scaling numerical variables                              
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])                       # Rescale numerical columns to [0, 1] range
print("A9 Results:")
print(df)

print("----------------------------------------------------------------------------------------------------------")