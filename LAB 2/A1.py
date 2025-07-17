import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
file_path = r"C:\\Users\\KIRAN\Downloads\\Lab Session Data.xlsx"  
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