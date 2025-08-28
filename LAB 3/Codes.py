import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the dataset
file_path = r"D:\\OneDrive - Amrita vishwa vidyapeetham\\SEM 5\\23CSE301 Machine Learning\\LAB\\Machine-Learning\\LAB 3\\20231225_dfall_obs_data_and_spectral_features_revision1_n469.csv"
df = pd.read_csv(file_path)

print("----------------------------------------------------------------------------------------------------------")

#A1. Intraclass and Interclass Analysis
class0 = 'Context2'                     # Target column
features = ['F1', 'F2', 'F3', 'F4']     # Feature columns
classes = df[class0].dropna().unique()  # Unique class labels
class1, class2 = classes[:2]            # First two classes
vecs1 = df[df[class0] == class1][features].values  # Features for class1
vecs2 = df[df[class0] == class2][features].values  # Features for class2
centroid1 = vecs1.mean(axis=0)           # Centroid for class1
centroid2 = vecs2.mean(axis=0)           # Centroid for class2
spread1 = vecs1.std(axis=0)              # Spread for class1
spread2 = vecs2.std(axis=0)              # Spread for class2
interclass_distance = np.linalg.norm(centroid1 - centroid2)     # Distance between centroids
print("A1 Answers: ")
print(f"Centroid for {class0}={class1}: {centroid1}")  
print(f"Centroid for {class0}={class2}: {centroid2}")  
print(f"Spread for {class0}={class1}: {spread1}")
print(f"Spread for {class0}={class2}: {spread2}") 
print(f"Interclass distance between {class1} and {class2}: {interclass_distance}")  # Print interclass distance

print("----------------------------------------------------------------------------------------------------------")

# A2. Feature density pattern analysis
feature = 'F1'
feature_data = df[feature].values                                                              # Use the first feature for density analysis
plt.figure(figsize=(10, 6))                                                             # Set figure size
plt.hist(feature_data, bins=20, color='skyblue', edgecolor='black', alpha=0.7)          # Histogram of feature data
plt.xlabel(f'{feature} values')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
mean_value = np.mean(feature_data)                                                      # Calculate mean of the feature
variance_value = np.var(feature_data)                                                   # Calculate variance of the feature
print("A2 Answers:")
print(f"Mean of {feature}: {mean_value:.4f}")
print(f"Variance of {feature}: {variance_value:.4f}")

print("----------------------------------------------------------------------------------------------------------")

# A3. Minkowski distance analysis
vec1 = df['F1'].values                                                                     # Select first vector
vec2 = df['F2'].values                                                                     # Select second vector 
def minkowski_distance(vec1, vec2, r):                                              
    return np.sum(np.abs(vec1 - vec2) ** r) ** (1/r)                            # Minkowski distance formula
r_values = range(1, 11)                                                         # Define range of r values
distances = [minkowski_distance(vec1, vec2, r) for r in r_values]               # Calculate distances for each r
plt.figure(figsize=(10, 6))                                                                             # Set figure size
plt.plot(r_values, distances, marker='o', linestyle='-', color='green', linewidth=2, markersize=8)      # Plot Minkowski distance vs r
plt.title('Minkowski Distance vs r (Order)')
plt.xlabel('r (Order of Minkowski Distance)')
plt.ylabel('Distance')
plt.grid(True)
plt.xticks(r_values)
plt.show()
print("A3 Answers:")
for r, distance in zip(r_values, distances):                                    # Print distance for each r value
    print(f"r={r}: {distance:.4f}")

print("----------------------------------------------------------------------------------------------------------")

# A4. Train-test split
X = df[['F1','F2','F3','F4']]
y = df['CallerSex']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)       # Split into train and test sets (70% train, 30% test)
print("A4 Answers:")      
print(f"Training features shape: {X_train.shape}")
print(f"Test features shape: {X_test.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test labels shape: {y_test.shape}")

print("----------------------------------------------------------------------------------------------------------")

# A5. kNN Classifier Training
k = 3                                                   # Set k value for kNN classifier
neigh = KNeighborsClassifier(n_neighbors=k)             # Train kNN classifier with k=3
neigh.fit(X_train, y_train)                             # Fit the model to the training data
print("A5 Answers:")
print(f"kNN classifier trained with k={k}")

print("----------------------------------------------------------------------------------------------------------")

# A6. Test accuracy using score method
test_accuracy = neigh.score(X_test, y_test)
print("A6 Answers:")                                                            
print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

print("----------------------------------------------------------------------------------------------------------")

# A7. Study prediction behavior using predict()
i = int(random.random()*X_test.shape[0])                        # Randomly select an index from the test set
test_vect = X_test.iloc[i]                                      # Get the corresponding test vector                    
predicted_class = neigh.predict([test_vect])                    # Predict the class for the selected test vector
print("A7 Answers:")
print(f"Test vector {i}: {test_vect}")
print(f"Predicted class: {predicted_class[0]}")
print(f"Actual class: {y_test.iloc[i]}")

print("----------------------------------------------------------------------------------------------------------")

# A8. Compare different k values (1 to 11)

k_values = range(1, 12)                             # Range of k values to test
train_accuracies = []                               # List to store training accuracies
test_accuracies = []                                # List to store test accuracies
print("A8 Answers:")    
print("k\tTrain Acc\tTest Acc\tDifference")
for k in k_values:  
    knn_k = KNeighborsClassifier(n_neighbors=k)                     # Create kNN classifier with current k value
    knn_k.fit(X_train, y_train)                                     # Fit the model to the training data
    train_acc = knn_k.score(X_train, y_train)                       # Calculate training accuracy
    test_acc = knn_k.score(X_test, y_test)                          # Calculate test accuracy
    train_accuracies.append(train_acc)                              # Store training accuracy
    test_accuracies.append(test_acc)                                # Store test accuracy
    print(f"{k}\t{train_acc:.4f}\t\t{test_acc:.4f}\t\t{train_acc - test_acc:.4f}")
plt.figure(figsize=(12, 6))                                                                         
plt.plot(k_values, train_accuracies, marker='o', label='Training Accuracy', linewidth=2)            # Plot training accuracies
plt.plot(k_values, test_accuracies, marker='s', label='Test Accuracy', linewidth=2)                 # Plot test accuracies
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.title('kNN Accuracy vs k Value')
plt.legend()
plt.grid(True)
plt.xticks(k_values)
plt.show()
best_k_index = np.argmax(test_accuracies)                                       # Find index of best k value based on test accuracy     
best_k = k_values[best_k_index]                                                 # Get the best k value
best_test_acc = test_accuracies[best_k_index]                                   # Get the best test accuracy
print(f"\nBest k value: {best_k} with test accuracy: {best_test_acc:.4f}")
print(f"\nComparison:")
print(f"k=1 (Nearest Neighbor): Train={train_accuracies[0]:.4f}, Test={test_accuracies[0]:.4f}")
print(f"k=3 (k-Nearest Neighbor): Train={train_accuracies[2]:.4f}, Test={test_accuracies[2]:.4f}")
if test_accuracies[0] > test_accuracies[2]:
    print("NN (k=1) performs better on test data")
elif test_accuracies[0] < test_accuracies[2]:
    print("kNN (k=3) performs better on test data")
else:
    print("NN and kNN perform equally on test data")

print("----------------------------------------------------------------------------------------------------------")

# A9. Confusion Matrix and Performance Metrics Analysis
df["class"] = df["CallerSex"].astype('category').cat.codes          # Convert 'CallerSex' into categorical codes for classification
X = df[['F1','F2','F3','F4']].values                                # Features (input variables)
y = df['class'].values                                              # Target variable (the class labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)  # Split data into train and test sets (70-30 split)
knn = KNeighborsClassifier(n_neighbors=5)                 # Initialize K-Nearest Neighbors classifier with 5 neighbors
knn.fit(X_train, y_train)                                 # Train the model on the training data
y_train_pred = knn.predict(X_train)                       # Predict class labels for the training set
y_test_pred = knn.predict(X_test)                         # Predict class labels for the test set
print("A9 Answers")
print("Training Confusion Matrix:") 
print(confusion_matrix(y_train, y_train_pred)) 
print("\nTest Confusion Matrix:") 
print(confusion_matrix(y_test, y_test_pred)) 
print("\nTraining Classification Report:") 
print(classification_report(y_train, y_train_pred))
print("\nTest Classification Report:")
print(classification_report(y_test, y_test_pred))  # Show detailed metrics for test data
train_acc = (y_train == y_train_pred).mean()                # Calculate training accuracy (correct predictions / total predictions)
test_acc = (y_test == y_test_pred).mean()                   # Calculate test accuracy
print(f"\nTraining Accuracy: {train_acc:.2f}")
print(f"Test Accuracy: {test_acc:.2f}")
if train_acc > 0.90 and (train_acc - test_acc) > 0.15:  # Check if model is overfitting
    print("\nObservation: Model is Overfitting.")
elif train_acc < 0.70 and test_acc < 0.70:  # Check if model is underfitting
    print("\nObservation: Model is Underfitting.")
else:  # Model is performing reasonably well
    print("\nObservation: Model is Regular Fit (good generalization).")

print("----------------------------------------------------------------------------------------------------------")