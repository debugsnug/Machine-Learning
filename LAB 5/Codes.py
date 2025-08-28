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
features = ['F1']                           # List of features to use for regression (only 'F1' in this case)
for feature in features:                    # Loop through each feature in the list
    print(f"\nRunning Single-Feature Linear Regression with feature: {feature}")    # Display the current feature being used
    X_single = df_clean[[feature]]                  # Select the feature 'F1' as the independent variable (X)
    y_single = df_clean['CallerAge']                # Set the target variable 'CallerAge' as the dependent variable (y)
    X_train_single, X_test_single, y_train_single, y_test_single = train_test_split(X_single, y_single, test_size=0.2, random_state=42)  # Split data into training and testing sets (80% train, 20% test)
    lin_reg = LinearRegression()                               # Initialize the linear regression model
    lin_reg.fit(X_train_single, y_train_single)                # Fit the model using the training data
    y_train_pred_single = lin_reg.predict(X_train_single)      # Predict the target for the training set
    y_test_pred_single = lin_reg.predict(X_test_single)        # Predict the target for the test set
    print(f"Single-Feature Regression Coefficient for {feature}: {lin_reg.coef_[0]:.6f}")  # Output the coefficient (slope) for the feature
    print(f"Intercept: {lin_reg.intercept_:.6f}")              # Output the intercept (y-axis cut-off point) of the model

# F1 is the strongest single predictor, with a strong inverse relationship with the caller's age.

print("-------------------------------------------------------------------------------------------------------------------------------")
# A2: Compute Evaluation Metrics
mse_train = mean_squared_error(y_train_single, y_train_pred_single)               # Calculate Mean Squared Error for training set
rmse_train = np.sqrt(mse_train)                                                   # Calculate Root Mean Squared Error for training set
mape_train = mean_absolute_percentage_error(y_train_single, y_train_pred_single)  # Calculate Mean Absolute Percentage Error for training set
r2_train = r2_score(y_train_single, y_train_pred_single)                          # Calculate R² score for training set

mse_test = mean_squared_error(y_test_single, y_test_pred_single)                # Calculate Mean Squared Error for test set
rmse_test = np.sqrt(mse_test)                                                   # Calculate Root Mean Squared Error for test set
mape_test = mean_absolute_percentage_error(y_test_single, y_test_pred_single)   # Calculate Mean Absolute Percentage Error for test set
r2_test = r2_score(y_test_single, y_test_pred_single)                           # Calculate R² score for test set

print("Training Set Metrics:")
print(f"MSE: {mse_train:.6f}, RMSE: {rmse_train:.6f}, MAPE: {mape_train:.6f}, R²: {r2_train:.6f}")  # Display training set metrics

print("Test Set Metrics:")
print(f"MSE: {mse_test:.6f}, RMSE: {rmse_test:.6f}, MAPE: {mape_test:.6f}, R²: {r2_test:.6f}")  # Display test set metrics

print("-------------------------------------------------------------------------------------------------------------------------------")
# A3: Multi-Feature Linear Regression
multi_features = ['F1','F2','F3','F4']                                          # List of selected features for the regression model
multi_features = [col for col in multi_features if col in df_clean.columns]     # Filter only the columns that exist in the dataframe
X_multi = df_clean[multi_features]                                              # Select features (X) from the dataframe
y_multi = df_clean['CallerAge']                                                 # Select the target variable (y) from the dataframe
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)  # Split data into training and test sets
multi_reg = LinearRegression()                                          # Initialize the Linear Regression model
multi_reg.fit(X_train_multi, y_train_multi)                             # Train the model using the training data
y_train_pred_multi = multi_reg.predict(X_train_multi)                   # Make predictions on the training data
y_test_pred_multi = multi_reg.predict(X_test_multi)                     # Make predictions on the test data
print(f"Multi-Feature Regression Coefficients: {multi_reg.coef_}")      # Display the regression coefficients
print(f"Feature names: {multi_features}")           # Print the selected feature names
print(f"Intercept: {multi_reg.intercept_:.6f}")     # Display the intercept value of the model

# Calculate and display training set metrics
mse_train_multi = mean_squared_error(y_train_multi, y_train_pred_multi)               # Calculate Mean Squared Error for training set
rmse_train_multi = np.sqrt(mse_train_multi)                                           # Calculate Root Mean Squared Error for training set
mape_train_multi = mean_absolute_percentage_error(y_train_multi, y_train_pred_multi)  # Calculate MAPE for training set
r2_train_multi = r2_score(y_train_multi, y_train_pred_multi)                          # Calculate R² score for training set

# Calculate and display test set metrics
mse_test_multi = mean_squared_error(y_test_multi, y_test_pred_multi)                # Calculate Mean Squared Error for test set
rmse_test_multi = np.sqrt(mse_test_multi)                                           # Calculate Root Mean Squared Error for test set
mape_test_multi = mean_absolute_percentage_error(y_test_multi, y_test_pred_multi)   # Calculate MAPE for test set
r2_test_multi = r2_score(y_test_multi, y_test_pred_multi)                           # Calculate R² score for test set

# Print out training set metrics
print("\nMulti-Feature Training Set Metrics:")  
print(f"MSE: {mse_train_multi:.6f}, RMSE: {rmse_train_multi:.6f}, MAPE: {mape_train_multi:.6f}, R²: {r2_train_multi:.6f}")  # Display training set metrics

# Print out test set metrics
print("Multi-Feature Test Set Metrics:")  
print(f"MSE: {mse_test_multi:.6f}, RMSE: {rmse_test_multi:.6f}, MAPE: {mape_test_multi:.6f}, R²: {r2_test_multi:.6f}")  # Display test set metrics


print("-------------------------------------------------------------------------------------------------------------------------------")
# A4: K-Means Clustering
clustering_features = (['F1', 'F2', 'F3', 'F4'])                                                    # Define a list of features for clustering
clustering_features = list(set([col for col in clustering_features if col in df_clean.columns]))    # Filter features that exist in the dataframe columns
X_cluster = df_clean[clustering_features].dropna()                                                  # Select the relevant features and drop any rows with missing values
print(f"Clustering with {len(clustering_features)} features: {clustering_features}")    # Print the number of features used for clustering
scaler = StandardScaler()                                       # Initialize a StandardScaler for feature scaling
X_scaled = scaler.fit_transform(X_cluster)                      # Scale the features to have zero mean and unit variance
X_train_cluster, X_test_cluster = train_test_split(X_scaled, test_size=0.2, random_state=42)  # Split the scaled data into training and testing sets (80% train, 20% test)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)                                     # Initialize the KMeans model with 4 clusters, random seed, and 10 initializations
clusters_train = kmeans.fit_predict(X_train_cluster)                              # Fit the KMeans model on the training data and predict the cluster labels
print(f"K-Means Cluster Centers shape: {kmeans.cluster_centers_.shape}")          # Print the shape of the cluster centers
print("Cluster labels distribution:", np.bincount(clusters_train))                # Print the distribution of the predicted cluster labels in the training set

print("-------------------------------------------------------------------------------------------------------------------------------")
# A5: Clustering Evaluation Metrics
silhouette = silhouette_score(X_train_cluster, clusters_train)                  # Compute the Silhouette Score to evaluate clustering quality
calinski_harabasz = calinski_harabasz_score(X_train_cluster, clusters_train)    # Compute the Calinski-Harabasz score (variance ratio criterion)
davies_bouldin = davies_bouldin_score(X_train_cluster, clusters_train)          # Compute the Davies-Bouldin Index to evaluate clustering compactness and separation
print(f"Silhouette Score: {silhouette:.4f}")                        # Print the Silhouette Score (higher values indicate better clustering)
print(f"Calinski-Harabasz Score: {calinski_harabasz:.4f}")          # Print the Calinski-Harabasz Score (higher values indicate better clustering)
print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")                # Print the Davies-Bouldin Index (lower values indicate better clustering)

print("-------------------------------------------------------------------------------------------------------------------------------")
# A6: Optimal k for Clustering
k_values = range(2, 10)     # Define the range of k (number of clusters) to evaluate
silhouette_scores = []      # List to store Silhouette Scores for each k
ch_scores = []              # List to store Calinski-Harabasz Scores for each k
db_scores = []              # List to store Davies-Bouldin Scores for each k

# Loop through each k value to evaluate clustering performance
for k in k_values:
    kmeans_k = KMeans(n_clusters=k, random_state=42, n_init=10)         # Initialize KMeans with k clusters
    labels_k = kmeans_k.fit_predict(X_train_cluster)                    # Fit the model and get the predicted cluster labels
    sil_score = silhouette_score(X_train_cluster, labels_k)             # Calculate Silhouette Score for the current k
    ch_score = calinski_harabasz_score(X_train_cluster, labels_k)       # Calculate Calinski-Harabasz Score for the current k
    db_score = davies_bouldin_score(X_train_cluster, labels_k)          # Calculate Davies-Bouldin Index for the current k
    silhouette_scores.append(sil_score)                         # Append Silhouette Score to the list
    ch_scores.append(ch_score)                                  # Append Calinski-Harabasz Score to the list
    db_scores.append(db_score)                                  # Append Davies-Bouldin Index to the list
    print(f"k={k}: Silhouette={sil_score:.4f}, CH={ch_score:.4f}, DB={db_score:.4f}")  # Print the evaluation metrics for the current k

# Plot the evaluation scores for different k values
plt.figure(figsize=(15, 5))              # Create a figure with a specific size
plt.subplot(1, 3, 1)                     # First subplot for Silhouette Score
plt.plot(k_values, silhouette_scores, marker='o', linewidth=2, markersize=8)  # Plot Silhouette Scores
plt.title('Silhouette Score vs k')          # Title of the first plot
plt.xlabel('Number of Clusters (k)')        # x-axis label
plt.ylabel('Silhouette Score')              # y-axis label
plt.grid(True, alpha=0.3)                   # Display grid with transparency
plt.subplot(1, 3, 2)                        # Second subplot for Calinski-Harabasz Score
plt.plot(k_values, ch_scores, marker='o', linewidth=2, markersize=8, color='orange')  # Plot Calinski-Harabasz Scores
plt.title('Calinski-Harabasz Score vs k')   # Title of the second plot
plt.xlabel('Number of Clusters (k)')        # x-axis label
plt.ylabel('CH Score')                      # y-axis label
plt.grid(True, alpha=0.3)                   # Display grid with transparency
plt.subplot(1, 3, 3)                        # Third subplot for Davies-Bouldin Index
plt.plot(k_values, db_scores, marker='o', linewidth=2, markersize=8, color='green')  # Plot Davies-Bouldin Scores
plt.title('Davies-Bouldin Index vs k')      # Title of the third plot
plt.xlabel('Number of Clusters (k)')        # x-axis label
plt.ylabel('DB Index')              # y-axis label
plt.grid(True, alpha=0.3)           # Display grid with transparency
plt.tight_layout()                  # Adjust layout to prevent overlap
plt.show()                          # Display the plots

# Identify optimal k values based on each evaluation metric
optimal_k_silhouette = k_values[np.argmax(silhouette_scores)]   # k with the highest Silhouette Score
optimal_k_ch = k_values[np.argmax(ch_scores)]                   # k with the highest Calinski-Harabasz Score
optimal_k_db = k_values[np.argmin(db_scores)]                   # k with the lowest Davies-Bouldin Index (ideal value is low)

# Print the optimal k values for each evaluation metric
print(f"\nOptimal k based on Silhouette Score: {optimal_k_silhouette}")   # Print optimal k for Silhouette Score
print(f"Optimal k based on Calinski-Harabasz Score: {optimal_k_ch}")      # Print optimal k for Calinski-Harabasz Score
print(f"Optimal k based on Davies-Bouldin Index: {optimal_k_db}")         # Print optimal k for Davies-Bouldin Index


print("-------------------------------------------------------------------------------------------------------------------------------")
# A7: Elbow Method for Optimal k
k_range = range(2, 20)      # Define the range of k values (number of clusters) to evaluate, from 2 to 19
distortions = []            # Initialize a list to store the inertia (distortion) for each k value
for k in k_range:           # Loop through each k value in the specified range
    kmeans_elbow = KMeans(n_clusters=k, random_state=42, n_init=10)  # Initialize KMeans model with k clusters
    kmeans_elbow.fit(X_train_cluster)                                # Fit the KMeans model to the training data
    distortions.append(kmeans_elbow.inertia_)                # Append the inertia (within-cluster sum of squared distances) to the distortions list
plt.figure(figsize=(10, 6))                                  # Create a new figure with specified size for the plot
plt.plot(k_range, distortions, marker='o', linewidth=2, markersize=8)  # Plot the inertia vs. number of clusters with a line and markers
plt.xlabel('Number of Clusters (k)')             # Set x-axis label: Number of clusters
plt.ylabel('Inertia (Distortion)')               # Set y-axis label: Inertia (Distortion)
plt.title('Elbow Method for Optimal k')          # Set the title of the plot
plt.grid(True, alpha=0.3)                        # Enable grid for the plot with some transparency
plt.show()                                       # Display the plot
