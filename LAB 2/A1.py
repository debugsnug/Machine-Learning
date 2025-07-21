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
try:
    df = pd.read_excel(xls, sheet_name="Purchase data")                           # Reading the "Purchase data" sheet into a DataFrame
    purchase_matrix = df.iloc[:, 1:4].values                                      # Extracting purchase matrix (quantities of products purchased) -> Matrix A
    purchase_amounts = df.iloc[:, 4].values.reshape(-1, 1)                        # Extracting purchase amounts (total cost per purchase)
    dimensionality = purchase_matrix.shape[1]                                     # Dimensionality of the vector space
    num_vectors = purchase_matrix.shape[0]                                        # Number of vectors (purchases)
    rank_A = np.linalg.matrix_rank(purchase_matrix)                               # Rank of the purchase matrix (A)
    purchase_matrix_pinv = np.linalg.pinv(purchase_matrix)                        # Computing Pseudo-Inverse
    product_costs = np.dot(purchase_matrix_pinv, purchase_amounts).flatten()      # Calculating product costs using pseudo-inverse
    print("A1 Answers:")
    print(f"    Dimensionality of the vector space: {dimensionality}")
    print(f"    Number of Vectors: {num_vectors}")
    print(f"    Rank of Matrix A: {rank_A}")
    print(f"    Product Costs: {product_costs}")
except FileNotFoundError:                                                         # Handles Excel file not found issues
    print(f"Error: File not found at {file_path}")
except ValueError:                                                                # Handles Excel sheet issues
    print("Error: Could not read specified sheet from Excel file.")

print("----------------------------------------------------------------------------------------------------------")

# A2: Classifying customers based on purchase amounts
df = pd.read_excel(xls, sheet_name="Purchase data")
df["Customer Class"] = np.where(df.iloc[:, 4] > 200, "RICH", "POOR")            # Classifying customers based on purchase amounts
df["Customer Class Label"] = (df["Customer Class"] == "RICH").astype(int)       # Converting class labels to binary (1 for RICH, 0 for POOR)
X = df.iloc[:, 1:4].values                                                      # Features: quantities of products purchased
y = df["Customer Class Label"].values                                           # Target: customer class labels (0 or 1)
clf = LogisticRegression()                                          # Initializing Logistic Regression classifier
clf.fit(X, y)                                                       # Fit the classifier to the data
preds = clf.predict(X)                                              # Predict on the same data (for demonstration)
df["Predicted Class"] = np.where(preds == 1, "RICH", "POOR")        # Adding predicted class to DataFrame
acc = accuracy_score(y, preds)                                      # Calculate accuracy of the classifier
print("A2 Answers:")
print(df[["Customer Class", "Predicted Class"]])
print(f"Classifier accuracy (on training data): {acc:.2f}")

print("----------------------------------------------------------------------------------------------------------")

#A3: Analyzing IRCTC stock price data
df = pd.read_excel(xls, sheet_name="IRCTC Stock Price")                       # Reading the "IRCTC Stock Price" sheet into a DataFrame
df["Date"] = pd.to_datetime(df["Date"])                                       # Converting 'Date' column to datetime format
df["Day"] = df["Date"].dt.day_name()                                          # Extracting day names from the 'Date' column
mean_price = statistics.mean(df["Price"])                                     # Calculating mean price
variance_price = statistics.variance(df["Price"])                             # Calculating variance of price
wednesday_mean = df[df["Day"] == "Wednesday"]["Price"].mean()                 # Calculating mean price for Wednesdays
april_mean = df[df["Date"].dt.month == 4]["Price"].mean()                     # Calculating mean price for April
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
ordinal_cols = ["age", "Condition", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]        # Ordinal columns based on domain knowledge
all_cat_cols = df.select_dtypes(include=['object']).columns.tolist()                # All categorical columns
nominal_cols = [col for col in all_cat_cols if col not in ordinal_cols]             # Nominal columns are those not in ordinal_cols
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()      # Numeric columns for analysis
print("\nA4 Results: ")
print("\nAttribute Types:")
for col in df.columns:
    if col in ordinal_cols:                                         # Check if column is ordinal or nominal     
        print(f"{col}: Ordinal")
    elif col in nominal_cols:
        print(f"{col}: Nominal")
print("\nEncoding Scheme:")
for col in ordinal_cols:
    if col in df.columns:
        print(f"{col}: Label Encoding (ordinal)")           
        df[col] = df[col].astype(str)                               # Convert ordinal columns to string for encoding
        df[col] = LabelEncoder().fit_transform(df[col])             # Label encoding for ordinal columns
for col in nominal_cols:
    print(f"{col}: One-Hot Encoding (nominal)")
df_encoded = pd.get_dummies(df, columns=nominal_cols, drop_first=True)          # One-hot encoding for nominal columns
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
print("\nEncoded DataFrame shape:", df_encoded.shape)                       # Encoded DataFrame shape
print(df_encoded.describe())                                                # Displaying descriptive statistics of the encoded DataFrame

print("----------------------------------------------------------------------------------------------------------")

# A5: Jaccard Coefficient and SMC (using only binary attributes)
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
df_filled = df.copy()                                                                        # Before cosine similarity calculation, fill missing values with median for numerical columns to avoid NaNs
valid_numeric_cols = [col for col in numeric_cols if col in df_filled.columns]               # Filter numeric columns that exist in the DataFrame 
for col in valid_numeric_cols:                                                               # Convert numeric columns to numeric type and fill NaNs with median 
    df_filled[col] = pd.to_numeric(df_filled[col], errors='coerce')                          # Fill NaNs with median
    df_filled[col] = df_filled[col].fillna(df_filled[col].median())                          # Reshape the first row into a 2D array
if len(valid_numeric_cols) == 0:                                                             # Check if there are valid numeric columns
    print("A6 Result: No valid numeric columns available for cosine similarity.")
else:                                                                                               
    vector1 = df_filled.loc[0, valid_numeric_cols].astype(float).values.reshape(1, -1)              # Reshape the first row into a 2D array
    vector2 = df_filled.loc[1, valid_numeric_cols].astype(float).values.reshape(1, -1)              # Reshape the second row into a 2D array
    result = cosine_similarity(vector1, vector2)[0][0]                                              # Calculate cosine similarity between the two vectors     
    print("A6 Result:", result)

print("----------------------------------------------------------------------------------------------------------")

# A7: Heatmap of JC, SMC, and COS similarities for first 20 vectors
df_subset = df.iloc[:20, :].copy()                                                                          # Subset the first 20 vectors
binary_cols = [col for col in df.columns if set(df_subset[col].dropna().unique()).issubset({0, 1})]         # Binary columns
numeric_cols_20 = [col for col in numeric_cols if col in df_subset.columns]                                 # Numeric columns in the subset
for col in numeric_cols_20:                                                                 # Convert numeric columns to numeric type and fill NaNs with median
    df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce')                         # Fill NaNs with median
    df_subset[col] = df_subset[col].fillna(df_subset[col].median())                         # Reshape the first row into a 2D array
JC_matrix = np.zeros((20, 20))                      # Initialize matrices for Jaccard Coefficient (JC)
SMC_matrix = np.zeros((20, 20))                     # Initialize Simple Matching Coefficient (SMC)
COS_matrix = np.zeros((20, 20))                     # Initialize Cosine Similarity (COS)
for i in range(20):                             # Loop through the first 20 vectors
    for j in range(20):                  
        v1 = df_subset.iloc[i][binary_cols].values if binary_cols else np.array([])         # Values of the i-th vector for binary columns
        v2 = df_subset.iloc[j][binary_cols].values if binary_cols else np.array([])         # Values of the j-th vector for binary columns
        if len(v1) > 0 and len(v2) > 0:                     # Check if both vectors have binary values
            f11 = np.sum((v1 == 1) & (v2 == 1))             # Count of matches where both are 1
            f00 = np.sum((v1 == 0) & (v2 == 0))             # Count of matches where both are 0
            f10 = np.sum((v1 == 1) & (v2 == 0))             # Count of matches where first is 1 and second is 0
            f01 = np.sum((v1 == 0) & (v2 == 1))             # Count of matches where first is 1 and second is 0
            denominator = (f01 + f10 + f11)                                             # Denominator for Jaccard Coefficient calculation
            JC_matrix[i, j] = f11 / denominator if denominator != 0 else 0              # Handles the case where all are 0
            SMC_matrix[i, j] = (f11 + f00) / (f00 + f01 + f10 + f11) if (f00 + f01 + f10 + f11) != 0 else 0         # Handles the case where all are 0
        else:
            JC_matrix[i, j] = np.nan                # If no binary values, set JC to NaN
            SMC_matrix[i, j] = np.nan               # If no binary values, set SMC to NaN
        v1_num = df_subset.iloc[i][numeric_cols_20].astype(float).values if numeric_cols_20 else np.array([])           # Values of the i-th vector for numeric columns
        v2_num = df_subset.iloc[j][numeric_cols_20].astype(float).values if numeric_cols_20 else np.array([])           # Values of the j-th vector for numeric columns
        if len(v1_num) > 0 and len(v2_num) > 0:                                                                         # Check if both vectors have numeric values 
            COS_matrix[i, j] = cosine_similarity(v1_num.reshape(1, -1), v2_num.reshape(1, -1))[0][0]                    # Calculate cosine similarity between the two numeric vectors
        else:
            COS_matrix[i, j] = np.nan               # If no numeric values, set COS to NaN
plt.figure(figsize=(24, 7))                                                     # Sets the figure size for the heatmaps
plt.subplot(1, 3, 1)                                                            
sns.heatmap(JC_matrix, annot=False, cmap='YlGnBu', vmin=0, vmax=1)              # Heatmap for Jaccard Coefficient (JC) 
plt.title('Jaccard Coefficient (JC)')                                           
plt.subplot(1, 3, 2)                                                            
sns.heatmap(SMC_matrix, annot=False, cmap='YlOrRd', vmin=0, vmax=1)             # Heatmap for Simple Matching Coefficient (SMC)
plt.title('Simple Matching Coefficient (SMC)')
plt.subplot(1, 3, 3)
sns.heatmap(COS_matrix, annot=False, cmap='coolwarm', vmin=0, vmax=1)           # Heatmap for Cosine Similarity (COS)
plt.title('Cosine Similarity (COS)')
plt.tight_layout()                                                              # Adjust layout to prevent overlap  
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

# Q1
df_purchase = pd.read_excel(xls, sheet_name="Purchase data")        # Reading the "Purchase data" sheet into a DataFrame
purchase_matrix_full = df_purchase.iloc[:, 1:4].values              # Original matrix (may not be square)
square_matrix1 = df_purchase.iloc[:3, 1:4].values                   # Creating first square matrix (3x3)
amounts1 = df_purchase.iloc[:3, 4].values.reshape(-1, 1)            # Vector of amounts (3x1)
square_matrix2 = df_purchase.iloc[-3:, 1:4].values                  # Creating second square matrix (3x3)
amounts2 = df_purchase.iloc[-3:, 4].values.reshape(-1, 1)           # Vector of amounts (3x1)

# --- Experiment A2: Logistic Regression on square matrices ---
# First square matrix
target1 = (amounts1.flatten() > 200).astype(int)                    # Creating target variable for classification (1 if amount > 200, else 0)
if len(np.unique(target1)) > 1:                                     # Check if there are at least two classes in the target variable
    clf1 = LogisticRegression()                                     # Initialize Logistic Regression classifier
    clf1.fit(square_matrix1, target1)                               # Fit the classifier to the first square matrix
    preds1 = clf1.predict(square_matrix1)                           # Predicting on the same data
    print("A2 with Square Matrix 1:")
    print("                     Predicted Classes:", preds1)
else:
    print("A2 with Square Matrix 1: Cannot fit Logistic Regression, only one class present:", np.unique(target1))
# Second square matrix
target2 = (amounts2.flatten() > 200).astype(int)            
if len(np.unique(target2)) > 1:
    clf2 = LogisticRegression()
    clf2.fit(square_matrix2, target2)
    preds2 = clf2.predict(square_matrix2)
    print("A2 with Square Matrix 2:")
    print("                     Predicted Classes:", preds2)
else:
    print("A2 with Square Matrix 2: Cannot fit Logistic Regression, only one class present:", np.unique(target2))

# --- Experiment A3: Product cost estimation using pseudo-inverse ---
# First square matrix
pinv1 = np.linalg.pinv(square_matrix1)                                  # Compute pseudo-inverse of the first square matrix
product_costs1 = np.dot(pinv1, amounts1).flatten()                      # Calculate product costs using pseudo-inverse
print("A3 with Square Matrix 1:")           
print("                     Product Costs:", product_costs1)
# Second square matrix
pinv2 = np.linalg.pinv(square_matrix2)                           # Compute pseudo-inverse of the second square matrix
product_costs2 = np.dot(pinv2, amounts2).flatten()               # Calculate product costs using pseudo-inverse
print("A3 with Square Matrix 2:")
print("                     Product Costs:", product_costs2)

# --- Compare with original purchase data matrix ---
pinv_full = np.linalg.pinv(purchase_matrix_full)                    # Compute pseudo-inverse of the full purchase matrix
amounts_full = df_purchase.iloc[:, 4].values.reshape(-1, 1)         # Vector of amounts for the full matrix
product_costs_full = np.dot(pinv_full, amounts_full).flatten()      # Calculate product costs using pseudo-inverse of the full matrix
print("A3 with Full Purchase Matrix:")
print("                     Product Costs:", product_costs_full)

print("Do X values (product costs) from square matrices match the full matrix?")
print("                     Matrix 1:", np.allclose(product_costs1, product_costs_full[:3]))                 # Check if product costs from first square matrix match the full matrix
print("                     Matrix 2:", np.allclose(product_costs2, product_costs_full[-3:]))                # Check if product costs from second square matrix match the full matrix

print("----------------------------------------------------------------------------------------------------------")

# Q2
df_thyroid = pd.read_excel(xls, sheet_name="thyroid0387_UCI")                       # Reading the "thyroid0387_UCI" sheet into a DataFrame
df_thyroid.replace('?', np.nan, inplace=True)                                       # Replaces '?' with NaN for proper handling of missing values   
df_thyroid = df_thyroid.infer_objects()                                             # Ensures proper type conversion
df_sample = df_thyroid.sample(n=20, random_state=42).reset_index(drop=True)         # Random sample of 20 rows

# --- A4: Data Exploration on the sample ---
ordinal_cols = ["age", "Condition", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]                    # List of ordinal columns
all_cat_cols = df_sample.select_dtypes(include=['object']).columns.tolist()                     # All categorical columns
nominal_cols = [col for col in all_cat_cols if col not in ordinal_cols]                         # Nominal columns are those not in ordinal_cols
numeric_cols = df_sample.select_dtypes(include=['float64', 'int64']).columns.tolist()           # Numeric columns for analysis
print("\nA4 Results (Random Sample):")
print("\nAttribute Types:")
for col in df_sample.columns:                                   # Check if column is ordinal or nominal
    if col in ordinal_cols:                                     # Ordinal columns are those in ordinal_cols
        print(f"{col}: Ordinal")                    
    elif col in nominal_cols:                                   # Nominal columns are those not in ordinal_cols
        print(f"{col}: Nominal")
print("\nEncoding Scheme:")
for col in ordinal_cols:                                        # Check if column is ordinal and apply label encoding
    if col in df_sample.columns:                                # Ordinal columns are encoded using Label Encoding
        print(f"{col}: Label Encoding (ordinal)")
        df_sample[col] = df_sample[col].astype(str)                         # Convert ordinal columns to string for encoding
        df_sample[col] = LabelEncoder().fit_transform(df_sample[col])       # Label encoding for ordinal columns
for col in nominal_cols:                                        # Nominal columns are encoded using One-Hot Encoding    
    print(f"{col}: One-Hot Encoding (nominal)")
df_encoded_sample = pd.get_dummies(df_sample, columns=nominal_cols, drop_first=True)        # One-hot encoding for nominal columns
print("\nNumeric Variable Ranges:")
for col in numeric_cols:                                                                
    print(f"{col}: min={df_sample[col].min()}, max={df_sample[col].max()}")             # Displaying min and max for numeric variables
print("\nMissing Values per Attribute:\n", df_sample.isnull().sum())    
print("\nOutlier Detection (IQR method):")
for col in numeric_cols:
    Q1 = df_sample[col].quantile(0.25)                                  # First quartile (25th percentile)
    Q3 = df_sample[col].quantile(0.75)                                  # Third quartile (75th percentile)
    IQR = Q3 - Q1                                                                                           # Interquartile range
    outliers = df_sample[(df_sample[col] < Q1 - 1.5 * IQR) | (df_sample[col] > Q3 + 1.5 * IQR)][col]        # Outliers are those below Q1 - 1.5*IQR or above Q3 + 1.5*IQR
    print(f"{col}: {len(outliers)} outliers")
print("\nMean and Std for Numeric Variables:")
for col in numeric_cols:
    print(f"{col}: mean={df_sample[col].mean()}, std={df_sample[col].std()}")           # Displaying mean and standard deviation for numeric variables
print("\nEncoded DataFrame shape:", df_encoded_sample.shape)
print(df_encoded_sample.describe())

# --- A5: Jaccard Coefficient and SMC (using only binary attributes) ---
binary_cols = [col for col in df_sample.columns if set(df_sample[col].dropna().unique()).issubset({0, 1})]          # Binary columns are those with only 0 and 1 values
v1 = df_sample.iloc[0][binary_cols].values                      # First binary vector
v2 = df_sample.iloc[1][binary_cols].values                      # Second binary vector
f11 = np.sum((v1 == 1) & (v2 == 1))                             # Count of matches where both are 1
f00 = np.sum((v1 == 0) & (v2 == 0))                             # Count of matches where both are 0
f10 = np.sum((v1 == 1) & (v2 == 0))                             # Count of matches where first is 1 and second is 0
f01 = np.sum((v1 == 0) & (v2 == 1))                             # Count of matches where first is 0 and second is 1
denominator = (f01 + f10 + f11)                                 # Denominator for Jaccard Coefficient calculation
JC = f11 / denominator if denominator != 0 else 0                                                # Handles the case where all are 0
SMC = (f11 + f00) / (f00 + f01 + f10 + f11) if (f00 + f01 + f10 + f11) != 0 else 0               # Handles the case where all are 0
print("\nA5 Results (Random Sample):")
print(f"        Jaccard Coefficient: {JC}, SMC: {SMC}")
print(f"        Number of binary attributes considered: {len(binary_cols)}")
if JC > SMC:        # Indicates more shared positive matches
    print("     JC is higher, indicating more shared positive matches. JC is appropriate when only positive matches matter.")
elif JC < SMC:      # Indicates more shared matches including negatives
    print("     SMC is higher, indicating both matches (0 and 1) are considered. SMC is appropriate when both matches are important.")
else:               # Indicates both are equal
    print("     JC and SMC are equal, indicating similar matching for this pair.")

# --- A6: Cosine Similarity Calculation ---
df_filled_sample = df_sample.copy()                                                                     # Before cosine similarity calculation, fill missing values with median for numerical columns to avoid NaNs
valid_numeric_cols = [col for col in numeric_cols if col in df_filled_sample.columns]                   # Filter numeric columns that exist in the DataFrame
for col in valid_numeric_cols:                                                                          # Convert numeric columns to numeric type and fill NaNs with median
    df_filled_sample[col] = pd.to_numeric(df_filled_sample[col], errors='coerce')                       # Fill NaNs with median
    df_filled_sample[col] = df_filled_sample[col].fillna(df_filled_sample[col].median())                # Reshape the first row into a 2D array
if len(valid_numeric_cols) == 0:                                                                        # Check if there are valid numeric columns
    print("\nA6 Result (Random Sample): No valid numeric columns available for cosine similarity.")
else:
    vector1 = df_filled_sample.loc[0, valid_numeric_cols].astype(float).values.reshape(1, -1)           # Reshape the first row into a 2D array
    vector2 = df_filled_sample.loc[1, valid_numeric_cols].astype(float).values.reshape(1, -1)           # Reshape the second row into a 2D array
    result = cosine_similarity(vector1, vector2)[0][0]                                                  # Calculate cosine similarity between the two vectors                            
    print("\nA6 Result (Random Sample):", result)

print("----------------------------------------------------------------------------------------------------------")

# Q3
df_marketing = pd.read_excel(xls, sheet_name="marketing_campaign")                          # Load the marketing campaign data
df_marketing.replace('?', np.nan, inplace=True)                                             # Replaces '?' with NaN for proper handling of missing values
df_marketing = df_marketing.infer_objects()                                                 # Ensures proper type conversion
df_sample_mkt = df_marketing.sample(n=20, random_state=42).reset_index(drop=True)           # Random sample of 20 rows

# --- A4: Data Exploration on the sample ---
ordinal_cols_mkt = []                                                                               # Specify ordinal columns if known, else leave empty
all_cat_cols_mkt = df_sample_mkt.select_dtypes(include=['object']).columns.tolist()                 # All categorical columns
nominal_cols_mkt = [col for col in all_cat_cols_mkt if col not in ordinal_cols_mkt]                 # Nominal columns are those not in ordinal_cols_mkt
numeric_cols_mkt = df_sample_mkt.select_dtypes(include=['float64', 'int64']).columns.tolist()       # Numeric columns for analysis
print("\nA4 Results (Random Sample - Marketing Campaign):")
print("\nAttribute Types:")
for col in df_sample_mkt.columns:                               # Check if column is ordinal or nominal
    if col in ordinal_cols_mkt:                                  
        print(f"{col}: Ordinal")
    elif col in nominal_cols_mkt:
        print(f"{col}: Nominal")
print("\nEncoding Scheme:")
for col in ordinal_cols_mkt:                                    # Check if column is ordinal and apply label encoding
    if col in df_sample_mkt.columns:                            
        print(f"{col}: Label Encoding (ordinal)")
        df_sample_mkt[col] = df_sample_mkt[col].astype(str)                             # Convert ordinal columns to string for encoding
        df_sample_mkt[col] = LabelEncoder().fit_transform(df_sample_mkt[col])           # Label encoding for ordinal columns
for col in nominal_cols_mkt:                                    # Check if column is nominal and apply one-hot encoding
    print(f"{col}: One-Hot Encoding (nominal)")
df_encoded_sample_mkt = pd.get_dummies(df_sample_mkt, columns=nominal_cols_mkt, drop_first=True)        # One-hot encoding for nominal columns
print("\nNumeric Variable Ranges:")
for col in numeric_cols_mkt:                                                                        # Check if column is numeric and print its range
    print(f"{col}: min={df_sample_mkt[col].min()}, max={df_sample_mkt[col].max()}")                
print("\nMissing Values per Attribute:\n", df_sample_mkt.isnull().sum())                        # Count missing values in each column
print("\nOutlier Detection (IQR method):")
for col in numeric_cols_mkt:                            # Check if column is numeric and apply IQR method for outlier detection
    Q1 = df_sample_mkt[col].quantile(0.25)                    # First quartile (25th percentile)
    Q3 = df_sample_mkt[col].quantile(0.75)                    # Third quartile (75th percentile)
    IQR = Q3 - Q1                                             # Interquartile range
    outliers = df_sample_mkt[(df_sample_mkt[col] < Q1 - 1.5 * IQR) | (df_sample_mkt[col] > Q3 + 1.5 * IQR)][col]        # Identifying outliers using IQR method
    print(f"{col}: {len(outliers)} outliers")                                               # Displaying number of outliers for each numeric variable
print("\nMean and Std for Numeric Variables:")
for col in numeric_cols_mkt:                                                                    # Check if column is numeric and print its mean and standard deviation
    print(f"{col}: mean={df_sample_mkt[col].mean()}, std={df_sample_mkt[col].std()}")           
print("\nEncoded DataFrame shape:", df_encoded_sample_mkt.shape)                                # Encoded DataFrame shape
print(df_encoded_sample_mkt.describe())                                                         # Displaying descriptive statistics of the encoded DataFrame

# --- A5: Jaccard Coefficient and SMC (using only binary attributes) ---
binary_cols_mkt = [col for col in df_sample_mkt.columns if set(df_sample_mkt[col].dropna().unique()).issubset({0, 1})]          # Binary columns
v1_mkt = df_sample_mkt.iloc[0][binary_cols_mkt].values                  # Values of the first row for binary columns
v2_mkt = df_sample_mkt.iloc[1][binary_cols_mkt].values                  # Values of the second row for binary columns
f11_mkt = np.sum((v1_mkt == 1) & (v2_mkt == 1))                         # Number of positions where both vectors are 1
f00_mkt = np.sum((v1_mkt == 0) & (v2_mkt == 0))                         # Count of matches where both are 0
f10_mkt = np.sum((v1_mkt == 1) & (v2_mkt == 0))                         # Count of matches where first is 1 and second is 0
f01_mkt = np.sum((v1_mkt == 0) & (v2_mkt == 1))                         # Count of matches where first is 0 and second is 1
denominator_mkt = (f01_mkt + f10_mkt + f11_mkt)                         # Denominator for Jaccard Coefficient calculation
JC_mkt = f11_mkt / denominator_mkt if denominator_mkt != 0 else 0                                                                       # Handles the case where all are 0
SMC_mkt = (f11_mkt + f00_mkt) / (f00_mkt + f01_mkt + f10_mkt + f11_mkt) if (f00_mkt + f01_mkt + f10_mkt + f11_mkt) != 0 else 0          # SMC calculation
print("\nA5 Results (Random Sample - Marketing Campaign):")
print(f"        Jaccard Coefficient: {JC_mkt}, SMC: {SMC_mkt}")
print(f"        Number of binary attributes considered: {len(binary_cols_mkt)}")
if JC_mkt > SMC_mkt:        # Indicates more shared positive matches
    print("     JC is higher, indicating more shared positive matches. JC is appropriate when only positive matches matter.")
elif JC_mkt < SMC_mkt:      # Indicates more shared negative matches
    print("     SMC is higher, indicating both matches (0 and 1) are considered. SMC is appropriate when both matches are important.")
else:                       # Indicates both are equal
    print("     JC and SMC are equal, indicating similar matching for this pair.")

# --- A6: Cosine Similarity Calculation ---
df_filled_sample_mkt = df_sample_mkt.copy()                                                                         # Create a copy of the original DataFrame
valid_numeric_cols_mkt = [col for col in numeric_cols_mkt if col in df_filled_sample_mkt.columns]                   # Filter numeric columns that exist in the DataFrame
for col in valid_numeric_cols_mkt:                                                                                  # Convert numeric columns to numeric type and fill NaNs with median
    df_filled_sample_mkt[col] = pd.to_numeric(df_filled_sample_mkt[col], errors='coerce')                        
    df_filled_sample_mkt[col] = df_filled_sample_mkt[col].fillna(df_filled_sample_mkt[col].median())             
if len(valid_numeric_cols_mkt) == 0:                                                                                                    # Check if there are valid numeric columns
    print("\nA6 Result (Random Sample - Marketing Campaign): No valid numeric columns available for cosine similarity.")
else:
    vector1_mkt = df_filled_sample_mkt.loc[0, valid_numeric_cols_mkt].astype(float).values.reshape(1, -1)               # Reshape the first row into a 2D array
    vector2_mkt = df_filled_sample_mkt.loc[1, valid_numeric_cols_mkt].astype(float).values.reshape(1, -1)               # Reshape the second row into a 2D array
    result_mkt = cosine_similarity(vector1_mkt, vector2_mkt)[0][0]                                                      # Calculate cosine similarity
    print("\nA6 Result (Random Sample - Marketing Campaign):", result_mkt)

print("----------------------------------------------------------------------------------------------------------")
