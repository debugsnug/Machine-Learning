import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
file_path = r"D:\\OneDrive - Amrita vishwa vidyapeetham\\SEM 5\\23CSE301 Machine Learning\\LAB\\Machine-Learning\\LAB 2\\Lab Session Data.xlsx"  
xls = pd.ExcelFile(file_path)

try:
    df = pd.read_excel(xls, sheet_name="Purchase data")
    purchase_matrix = df.iloc[:, 1:4].values  
    purchase_amounts = df.iloc[:, 4].values.reshape(-1, 1)  
    dimensionality = purchase_matrix.shape[1]
    num_vectors = purchase_matrix.shape[0]
    rank_A = np.linalg.matrix_rank(purchase_matrix)
    purchase_matrix_pinv = np.linalg.pinv(purchase_matrix)
    product_costs = np.dot(purchase_matrix_pinv, purchase_amounts).flatten()  
    print("A1 Results:")
    print(f"Dimensionality: {dimensionality}")
    print(f"Number of Vectors: {num_vectors}")
    print(f"Rank of A: {rank_A}")
    print(f"Product Costs: {product_costs}")
    
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    
except ValueError:  # Catches potential Excel sheet issues
    print("Error: Could not read specified sheet from Excel file.")
    
print(f"Model Vector X: {product_costs}")

df = pd.read_excel(xls, sheet_name="Purchase data")
df["Customer Class"] = np.where(df.iloc[:, 4] > 200, "RICH", "POOR")  
print("A2 Result:")
print(df[["Customer Class"]])

df = pd.read_excel(xls, sheet_name="IRCTC Stock Price")
df["Date"] = pd.to_datetime(df["Date"])
df["Day"] = df["Date"].dt.day_name()

mean_price = statistics.mean(df["Price"])
variance_price = statistics.variance(df["Price"])
wednesday_mean = df[df["Day"] == "Wednesday"]["Price"].mean()  
april_mean = df[df["Date"].dt.month == 4]["Price"].mean()  
prob_loss = (df["Chg%"] < 0).mean()
prob_profit_wed = df[(df["Day"] == "Wednesday") & (df["Chg%"] > 0)]["Chg%"].count() / df[df["Day"] == "Wednesday"]["Chg%"].count()

print("A3 Results:")
print(f"Mean Price: {mean_price}")
print(f"Variance Price: {variance_price}")
print(f"Wednesday Mean Price: {wednesday_mean}")
print(f"April Mean Price: {april_mean}")
print(f"Probability of Loss: {prob_loss}")
print(f"Probability of Profit on Wednesday: {prob_profit_wed}")

plt.figure(figsize=(10, 5))
sns.scatterplot(x=df["Day"], y=df["Chg%"])
plt.xlabel("Day of the Week")  # axis labels
plt.ylabel("Change %")
plt.xticks(rotation=45)
plt.title("Change % vs. Day of the Week")
plt.tight_layout() #  prevents labels from overlapping
plt.show()

df = pd.read_excel(xls, sheet_name="thyroid0387_UCI")
df.replace('?', np.nan, inplace=True)
df = df.infer_objects()  # Ensures proper type conversion
missing_values = df.isnull().sum()

# Converts categorical columns to string for Label Encoding 
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].astype(str)  # Converts to string
    df[col] = LabelEncoder().fit_transform(df[col])

print("A4 Results:")
print(df.describe())
print("Missing Values:\n", missing_values)

vector1 = df.iloc[0, :].values
vector2 = df.iloc[1, :].values
f11 = np.sum((vector1 == 1) & (vector2 == 1))
f00 = np.sum((vector1 == 0) & (vector2 == 0))
f10 = np.sum((vector1 == 1) & (vector2 == 0))
f01 = np.sum((vector1 == 0) & (vector2 == 1))

# Check for division by zero
denominator = (f01 + f10 + f11)
JC = f11 / denominator if denominator != 0 else 0  # Handles the case where all are 0
SMC = (f11 + f00) / (f00 + f01 + f10 + f11) if (f00 + f01 + f10 + f11) != 0 else 0

print("A5 Results:")
print(f"Jaccard Coefficient: {JC}, SMC: {SMC}")

vector1 = df.iloc[0, :].values.reshape(1, -1)
vector2 = df.iloc[1, :].values.reshape(1, -1)
result = cosine_similarity(vector1, vector2)[0][0]
print("A6 Result:", result)

df_subset = df.iloc[:20, :]  # Uses a subset for better visualization
similarity_matrix = np.zeros((20, 20))

for i in range(20):
    for j in range(20):
        if i != j:
            similarity_matrix[i, j] = np.linalg.norm(df_subset.iloc[i] - df_subset.iloc[j])  # Euclidean distance

plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, annot=False, cmap='coolwarm')  # (annot=False) for cleaner heatmap
plt.title("Heatmap of Euclidean Distances (Dissimilarity)")
plt.tight_layout()
plt.show()

df = pd.read_excel(xls, sheet_name="thyroid0387_UCI")
df.replace('?', np.nan, inplace=True)
df = df.infer_objects()

for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

print("A8 Results:")
print(df)


categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print("A7 Results:")
print(df)