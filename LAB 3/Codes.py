import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the elephant call dataset
dataset = pd.read_csv(r"..\Dataset\\20231225_dfall_obs_data_and_spectral_features_revision1_n469.csv")

# For demonstration, let's use 'Call_Type' as the class label and a few spectral features as features
# We'll use only rows where Call_Type is not null and select two most frequent classes
dataset = dataset.dropna(subset=['Call_Type'])
top_classes = dataset['Call_Type'].value_counts().index[:2]
filtered = dataset[dataset['Call_Type'].isin(top_classes)]

# Select features (example: 'sprsMed', 'sprsMbw', 'sprsEqbw', 'sprsMc')
features = ['sprsMed', 'sprsMbw', 'sprsEqbw', 'sprsMc']
X = filtered[features].values
y = filtered['Call_Type'].values

# Split data by class
class_labels = np.unique(y)
if len(class_labels) < 2:
    print("Not enough classes found for comparison. Please check your dataset or filtering criteria.")
else:
    class_0_data = X[y == class_labels[0]]
    class_1_data = X[y == class_labels[1]]

    # Calculate centroids (means)
    centroid_0 = np.mean(class_0_data, axis=0)
    centroid_1 = np.mean(class_1_data, axis=0)

    # Calculate spreads (standard deviations)
    spread_0 = np.std(class_0_data, axis=0)
    spread_1 = np.std(class_1_data, axis=0)

    # Calculate Euclidean distance between centroids
    distance_between_centroids = np.linalg.norm(centroid_0 - centroid_1)

    # Print results
    print(f"Class labels: {class_labels}")
    print(f"Centroid for Class {class_labels[0]}: {centroid_0}")
    print(f"Centroid for Class {class_labels[1]}: {centroid_1}")
    print(f"Spread for Class {class_labels[0]}: {spread_0}")
    print(f"Spread for Class {class_labels[1]}: {spread_1}")
    print(f"Distance between centroids: {distance_between_centroids}")

    # Optional: visualize in 2D using first two features
    plt.figure(figsize=(8,6))
    plt.scatter(class_0_data[:,0], class_0_data[:,1], label=f'Class {class_labels[0]}', alpha=0.6)
    plt.scatter(class_1_data[:,0], class_1_data[:,1], label=f'Class {class_labels[1]}', alpha=0.6)
    plt.scatter(centroid_0[0], centroid_0[1], marker='X', color='red', s=100, label=f'Centroid {class_labels[0]}')
    plt.scatter(centroid_1[0], centroid_1[1], marker='X', color='blue', s=100, label=f'Centroid {class_labels[1]}')
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title('Class Data and Centroids')
    plt.legend()
    plt.grid(True)
    plt.show()

print("----------------------------------------------------------------------------------------------------------")

#A2.
feature = 'max'
data = dataset[feature].values  # Extract the feature data

# Plot the histogram
plt.figure(figsize=(10, 6))
plt.hist(data, bins=20, color='skyblue', edgecolor='black', alpha=0.7)  # 20 bins for histogram
plt.title(f'Histogram of {feature}')
plt.xlabel(f'{feature} values')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Calculate the mean and variance
mean_value = np.mean(data)
variance_value = np.var(data)

print(f"Mean of {feature}: {mean_value}")
print(f"Variance of {feature}: {variance_value}")

print("----------------------------------------------------------------------------------------------------------")

#A3.
# Choose two feature vectors 
vec1 = dataset[['max', 'distance_to_event']].iloc[1].values 
vec2 = dataset[['max', 'distance_to_event']].iloc[50].values  

# Function to compute Minkowski distance
def minkowski_distance(vec1, vec2, r):
    return np.sum(np.abs(vec1 - vec2) ** r) ** (1/r)

# List to store the distances
distances = []

# Calculate Minkowski distance for r from 1 to 10
for r in range(1, 11):
    distance = minkowski_distance(vec1, vec2, r)
    distances.append(distance)

# Plotting the Minkowski distances
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), distances, marker='o', linestyle='-', color='b')
plt.title('Minkowski Distance vs r (from 1 to 10)')
plt.xlabel('r (Order of Minkowski Distance)')
plt.ylabel('Distance')
plt.grid(True)
plt.xticks(range(1, 11))
plt.show()

# Print distances for reference
for r, distance in zip(range(1, 11), distances):
    print(f"Minkowski distance (r={r}): {distance}")

print("----------------------------------------------------------------------------------------------------------")

#A4. 
data = dataset.copy()
#data["class"] = create_intensity_classes(data)

class_1_and_2 = data[data['class'].isin([1, 2])]

# Extract features and labels again after filtering
X = class_1_and_2[['max', 'distance_to_event']].values
y = class_1_and_2['class'].values

# Split the data into training and test sets (70% train, 30% test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Check the shapes of the resulting datasets
# print("Training features shape:", X_train.shape)
#print("Test features shape:", X_test.shape)
#print("Training labels shape:", y_train.shape)
#print("Test labels shape:", y_test.shape)

print("----------------------------------------------------------------------------------------------------------")

#A5.

# k=3
#neigh = KNeighborsClassifier(n_neighbors=k)
#neigh.fit(X_train,y_train)
