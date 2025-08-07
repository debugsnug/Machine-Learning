import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

file_path = r"D:\\OneDrive - Amrita vishwa vidyapeetham\\SEM 5\\23CSE301 Machine Learning\\LAB\\Machine-Learning\\LAB 3\\20231225_dfall_obs_data_and_spectral_features_revision1_n469.csv"
df = pd.read_csv(file_path)

numerical_cols = ['start', 'end', 'Distance', 'Quality', 'CallerAge', 'Elicitor1Age', 'CallerBinAge', 'ElicitorBinAge', 'sprsMed', 'sprsMbw', 'sprsEqbw', 'sprsMc']     # List of numerical columns to be used in analysis (excluding categorical and ID columns)
v_cols = [f'V{i}' for i in range(1, 75)]                                                                                        # Add V columns (spectral features)
numerical_cols.extend(v_cols)                                                                                                   # Add V columns to the list of numerical columns
formant_cols = ['Fw1', 'Fw2', 'Fw3', 'Fw4', 'Mw1', 'Mw2', 'Mw3', 'Mw4', 'F1', 'F2', 'F3', 'F4', 'M1', 'M2', 'M3', 'M4']         # Add formant columns (spectral features)
numerical_cols.extend(formant_cols)                                                                                             # Add formant columns to the list of numerical columns
numerical_cols = [col for col in numerical_cols if col in df.columns]               # Filter out columns that are not present in the dataframe
df_clean = df[numerical_cols].dropna()

print("-------------------------------------------------------------------------------------------------------------------------------")
# A1: Single-Feature Linear Regression
df_clean['duration'] = df_clean['end'] - df_clean['start']                  # 
X_single = df_clean[['duration']]
y_single = df_clean['sprsMed']
X_train_single, X_test_single, y_train_single, y_test_single = train_test_split(X_single, y_single, test_size=0.2, random_state=42)
lin_reg = LinearRegression()
lin_reg.fit(X_train_single, y_train_single)
y_train_pred_single = lin_reg.predict(X_train_single)
y_test_pred_single = lin_reg.predict(X_test_single)
print(f"Single-Feature Regression Coefficient: {lin_reg.coef_[0]:.6f}")
print(f"Intercept: {lin_reg.intercept_:.6f}")

print("-------------------------------------------------------------------------------------------------------------------------------")
# A2: Compute Evaluation Metrics
mse_train = mean_squared_error(y_train_single, y_train_pred_single)
rmse_train = np.sqrt(mse_train)
mape_train = mean_absolute_percentage_error(y_train_single, y_train_pred_single)
r2_train = r2_score(y_train_single, y_train_pred_single)
mse_test = mean_squared_error(y_test_single, y_test_pred_single)
rmse_test = np.sqrt(mse_test)
mape_test = mean_absolute_percentage_error(y_test_single, y_test_pred_single)
r2_test = r2_score(y_test_single, y_test_pred_single)
print("Training Set Metrics:")
print(f"MSE: {mse_train:.6f}, RMSE: {rmse_train:.6f}, MAPE: {mape_train:.6f}, R²: {r2_train:.6f}")
print("Test Set Metrics:")
print(f"MSE: {mse_test:.6f}, RMSE: {rmse_test:.6f}, MAPE: {mape_test:.6f}, R²: {r2_test:.6f}")

print("-------------------------------------------------------------------------------------------------------------------------------")
# A3: Multi-Feature Linear Regression
multi_features = ['duration', 'Distance', 'CallerAge', 'sprsMbw', 'sprsEqbw']
multi_features = [col for col in multi_features if col in df_clean.columns]
X_multi = df_clean[multi_features]
y_multi = df_clean['sprsMed']
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)
multi_reg = LinearRegression()
multi_reg.fit(X_train_multi, y_train_multi)
y_train_pred_multi = multi_reg.predict(X_train_multi)
y_test_pred_multi = multi_reg.predict(X_test_multi)
print(f"Multi-Feature Regression Coefficients: {multi_reg.coef_}")
print(f"Feature names: {multi_features}")
print(f"Intercept: {multi_reg.intercept_:.6f}")
mse_train_multi = mean_squared_error(y_train_multi, y_train_pred_multi)
rmse_train_multi = np.sqrt(mse_train_multi)
mape_train_multi = mean_absolute_percentage_error(y_train_multi, y_train_pred_multi)
r2_train_multi = r2_score(y_train_multi, y_train_pred_multi)
mse_test_multi = mean_squared_error(y_test_multi, y_test_pred_multi)
rmse_test_multi = np.sqrt(mse_test_multi)
mape_test_multi = mean_absolute_percentage_error(y_test_multi, y_test_pred_multi)
r2_test_multi = r2_score(y_test_multi, y_test_pred_multi)
print("\nMulti-Feature Training Set Metrics:")
print(f"MSE: {mse_train_multi:.6f}, RMSE: {rmse_train_multi:.6f}, MAPE: {mape_train_multi:.6f}, R²: {r2_train_multi:.6f}")
print("Multi-Feature Test Set Metrics:")
print(f"MSE: {mse_test_multi:.6f}, RMSE: {rmse_test_multi:.6f}, MAPE: {mape_test_multi:.6f}, R²: {r2_test_multi:.6f}")

print("-------------------------------------------------------------------------------------------------------------------------------")
# A4: K-Means Clustering
clustering_features = [col for col in multi_features if col != 'sprsMed']
clustering_features.extend(['sprsMbw', 'sprsEqbw', 'sprsMc'])
clustering_features = list(set([col for col in clustering_features if col in df_clean.columns]))
X_cluster = df_clean[clustering_features].dropna()
print(f"Clustering with {len(clustering_features)} features: {clustering_features}")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)
X_train_cluster, X_test_cluster = train_test_split(X_scaled, test_size=0.2, random_state=42)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters_train = kmeans.fit_predict(X_train_cluster)
print(f"K-Means Cluster Centers shape: {kmeans.cluster_centers_.shape}")
print("Cluster labels distribution:", np.bincount(clusters_train))

print("-------------------------------------------------------------------------------------------------------------------------------")
# A5: Clustering Evaluation Metrics
silhouette = silhouette_score(X_train_cluster, clusters_train)
calinski_harabasz = calinski_harabasz_score(X_train_cluster, clusters_train)
davies_bouldin = davies_bouldin_score(X_train_cluster, clusters_train)
print(f"Silhouette Score: {silhouette:.4f}")
print(f"Calinski-Harabasz Score: {calinski_harabasz:.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")

print("-------------------------------------------------------------------------------------------------------------------------------")
# A6: Optimal k for Clustering
k_values = range(2, 10)
silhouette_scores = []
ch_scores = []
db_scores = []
for k in k_values:
    kmeans_k = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = kmeans_k.fit_predict(X_train_cluster)
    sil_score = silhouette_score(X_train_cluster, labels_k)
    ch_score = calinski_harabasz_score(X_train_cluster, labels_k)
    db_score = davies_bouldin_score(X_train_cluster, labels_k)
    silhouette_scores.append(sil_score)
    ch_scores.append(ch_score)
    db_scores.append(db_score)
    print(f"k={k}: Silhouette={sil_score:.4f}, CH={ch_score:.4f}, DB={db_score:.4f}")
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(k_values, silhouette_scores, marker='o', linewidth=2, markersize=8)
plt.title('Silhouette Score vs k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True, alpha=0.3)
plt.subplot(1, 3, 2)
plt.plot(k_values, ch_scores, marker='o', linewidth=2, markersize=8, color='orange')
plt.title('Calinski-Harabasz Score vs k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('CH Score')
plt.grid(True, alpha=0.3)
plt.subplot(1, 3, 3)
plt.plot(k_values, db_scores, marker='o', linewidth=2, markersize=8, color='green')
plt.title('Davies-Bouldin Index vs k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('DB Index')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
optimal_k_silhouette = k_values[np.argmax(silhouette_scores)]
optimal_k_ch = k_values[np.argmax(ch_scores)]
optimal_k_db = k_values[np.argmin(db_scores)]
print(f"\nOptimal k based on Silhouette Score: {optimal_k_silhouette}")
print(f"Optimal k based on Calinski-Harabasz Score: {optimal_k_ch}")
print(f"Optimal k based on Davies-Bouldin Index: {optimal_k_db}")

print("-------------------------------------------------------------------------------------------------------------------------------")
# A7: Elbow Method for Optimal k
k_range = range(2, 20)
distortions = []
for k in k_range:
    kmeans_elbow = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_elbow.fit(X_train_cluster)
    distortions.append(kmeans_elbow.inertia_)
plt.figure(figsize=(10, 6))
plt.plot(k_range, distortions, marker='o', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Distortion)')
plt.title('Elbow Method for Optimal k')
plt.grid(True, alpha=0.3)
plt.show()
def calculate_elbow_point(k_values, distortions):
    k_norm = np.array(k_values) / max(k_values)
    dist_norm = np.array(distortions) / max(distortions)
    n_points = len(k_values)
    all_coord = np.vstack((k_norm, dist_norm)).T
    first_point = all_coord[0]
    line_vec = all_coord[-1] - all_coord[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))    
    vec_from_first = all_coord - first_point
    scalar_product = np.sum(vec_from_first * line_vec_norm, axis=1)
    vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    dist_to_line = np.sqrt(np.sum(vec_to_line**2, axis=1))
    elbow_idx = np.argmax(dist_to_line)
    return k_values[elbow_idx]
optimal_k_elbow = calculate_elbow_point(list(k_range), distortions)
print(f"Optimal k based on Elbow Method: {optimal_k_elbow}")
print("Summary of Results")
print("Regression Results:")
print(f"  Single-feature R²: {r2_test:.4f}")
print(f"  Multi-feature R²: {r2_test_multi:.4f}")
print("\nClustering Results:")
print(f"  Best k (Silhouette): {optimal_k_silhouette}")
print(f"  Best k (Calinski-Harabasz): {optimal_k_ch}")
print(f"  Best k (Davies-Bouldin): {optimal_k_db}")
print(f"  Best k (Elbow Method): {optimal_k_elbow}")
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(X_test_single, y_test_single, alpha=0.6, label='Actual')
plt.scatter(X_test_single, y_test_pred_single, alpha=0.6, label='Predicted')
plt.xlabel('Duration')
plt.ylabel('Spectral Median')
plt.title(f'Single-Feature Regression (R² = {r2_test:.3f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.subplot(1, 2, 2)
plt.scatter(y_test_multi, y_test_pred_multi, alpha=0.6)
plt.plot([y_test_multi.min(), y_test_multi.max()], [y_test_multi.min(), y_test_multi.max()], 'r--', lw=2)
plt.xlabel('Actual Spectral Median')
plt.ylabel('Predicted Spectral Median')
plt.title(f'Multi-Feature Regression (R² = {r2_test_multi:.3f})')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("-------------------------------------------------------------------------------------------------------------------------------")
