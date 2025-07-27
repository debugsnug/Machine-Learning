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
dataset = pd.read_csv(file_path)

print("----------------------------------------------------------------------------------------------------------")

#A1. Intraclass and Interclass Analysis
class_counts = dataset['Context2'].value_counts()       # Two most frequent classes
class_1 = class_counts.index[0]                         
class_2 = class_counts.index[1]
binary_data = dataset[dataset['Context2'].isin([class_1, class_2])].copy()      # Filter dataset for these two classes
features = ['F1', 'F2']                                                     # Use F1 and F2 as features, but only keep rows where both are non-NaN
clean_binary_data = binary_data.dropna(subset=features)                     # If no rows left, try alternative features
if len(clean_binary_data) == 0:
    features = ['sprsMed', 'sprsMbw']                                       # Try alternative features like sprsMed, sprsMbw which seem to have values
    clean_binary_data = binary_data.dropna(subset=features)                 
X = clean_binary_data[features].values                  # Extract feature values
y = clean_binary_data['Context2'].values                # Extract target values
class_1_data = X[y == class_1]                          # Separate data by class
class_2_data = X[y == class_2]                          
# Calculate centroids (means)
centroid_1 = np.mean(class_1_data, axis=0)
centroid_2 = np.mean(class_2_data, axis=0)
# Calculate spreads (standard deviations)
spread_1 = np.std(class_1_data, axis=0)
spread_2 = np.std(class_2_data, axis=0)
# Calculate distance between centroids
distance_between_centroids = np.linalg.norm(centroid_1 - centroid_2)
print("A2 Answers:")
print(f"Centroid for {class_1}: [{centroid_1[0]:.4f}, {centroid_1[1]:.4f}]")
print(f"Centroid for {class_2}: [{centroid_2[0]:.4f}, {centroid_2[1]:.4f}]")
print(f"Spread for {class_1}: [{spread_1[0]:.4f}, {spread_1[1]:.4f}]")
print(f"Spread for {class_2}: [{spread_2[0]:.4f}, {spread_2[1]:.4f}]")
print(f"Distance between centroids: {distance_between_centroids:.4f}")
# Visualize the classes
plt.figure(figsize=(10, 6))
plt.scatter(class_1_data[:, 0], class_1_data[:, 1], color='red', label=f'{class_1}', alpha=0.6)             # Scatter plot for class 1
plt.scatter(class_2_data[:, 0], class_2_data[:, 1], color='blue', label=f'{class_2}', alpha=0.6)            # Scatter plot for class 2
# Plot centroids
plt.scatter(centroid_1[0], centroid_1[1], color='red', marker='X', s=200,               
           edgecolor='black', label=f'Centroid {class_1}')                                          
plt.scatter(centroid_2[0], centroid_2[1], color='blue', marker='X', s=200, 
           edgecolor='black', label=f'Centroid {class_2}')
plt.xlabel(f'{features[0]}')
plt.ylabel(f'{features[1]}')
plt.title('Elephant Call Classes with Centroids')
plt.legend()
plt.grid(True)
plt.show()

print("----------------------------------------------------------------------------------------------------------")

# A2. Feature density pattern analysis
feature_name = features[0]                                                              # Use the first feature for density analysis
feature_data = clean_binary_data[feature_name].values                                   # Extract feature values
plt.figure(figsize=(10, 6))                                                             # Set figure size
plt.hist(feature_data, bins=20, color='skyblue', edgecolor='black', alpha=0.7)          # Histogram of feature data
plt.xlabel(f'{feature_name} values')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
mean_value = np.mean(feature_data)                                                      # Calculate mean of the feature
variance_value = np.var(feature_data)                                                   # Calculate variance of the feature
print("A2 Answers:")
print(f"Mean of {feature_name}: {mean_value:.4f}")
print(f"Variance of {feature_name}: {variance_value:.4f}")

print("----------------------------------------------------------------------------------------------------------")

# A3. Minkowski distance analysis
vec1 = X[0]                                                                     # Select first vector
vec2 = X[5] if len(X) > 5 else X[1]                                             # Select second vector (5th if exists, else 2nd)
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)       # Split into train and test sets (70% train, 30% test)
print("A4 Answers:")      
print(f"Training features shape: {X_train.shape}")
print(f"Test features shape: {X_test.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test labels shape: {y_test.shape}")
print(f"\nTraining set class distribution:")
train_unique, train_counts = np.unique(y_train, return_counts=True)                              # Count unique classes in training set
for class_name, count in zip(train_unique, train_counts):
    print(f"  {class_name}: {count}")                                                            # Print class distribution
print(f"\nTest set class distribution:")
test_unique, test_counts = np.unique(y_test, return_counts=True)
for class_name, count in zip(test_unique, test_counts):
    print(f"  {class_name}: {count}")

print("----------------------------------------------------------------------------------------------------------")

# A5. kNN Classifier Training
k = 3                                                   # Set k value for kNN classifier
neigh = KNeighborsClassifier(n_neighbors=k)             # Train kNN classifier with k=3
neigh.fit(X_train, y_train)                             # Fit the model to the training data
print("A5 Answers:")
print(f"kNN classifier trained with k={k}")
print(f"Features used: {features}")

print("----------------------------------------------------------------------------------------------------------")

# A6. Test accuracy using score method
test_accuracy = neigh.score(X_test, y_test)
print("A6 Answers:")                                                            
print(f"Test accuracy using neigh.score(): {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

print("----------------------------------------------------------------------------------------------------------")

# A7. Study prediction behavior using predict()
i = int(random.random()*X_test.shape[0])                    # Randomly select an index from the test set
test_vect = X_test[i]                                       # Get the corresponding test vector                    
predicted_class = neigh.predict([test_vect])                # Predict the class for the selected test vector
print("A7 Answers:")
print(f"Test vector {i}: {test_vect}")
print(f"Predicted class: {predicted_class[0]}")
print(f"Actual class: {y_test[i]}")

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
print("A9 Answers:")
best_knn = KNeighborsClassifier(n_neighbors=best_k)                 # Create kNN classifier with best k value
best_knn.fit(X_train, y_train)                                      # Fit the model to the training data
y_train_pred = best_knn.predict(X_train)                            # Predict on training set
y_test_pred = best_knn.predict(X_test)                              # Predict on test set   
train_cm = confusion_matrix(y_train, y_train_pred)                  # Train confusion matrix
test_cm = confusion_matrix(y_test, y_test_pred)                     # Test confusion matrix
print(f"Using best k = {best_k}")
print("\nTraining Set Confusion Matrix:")
print(f"Classes: {best_knn.classes_}")
print(train_cm)
print("\nTest Set Confusion Matrix:")
print(f"Classes: {best_knn.classes_}")
print(test_cm)
# Calculate performance metrics
average_method = 'binary' if len(np.unique(y)) == 2 else 'weighted'             # Use binary for 2 classes, weighted for more than 2
# Training metrics
train_precision = precision_score(y_train, y_train_pred, average=average_method, pos_label=class_1)         # Precision
train_recall = recall_score(y_train, y_train_pred, average=average_method, pos_label=class_1)               # Recall
train_f1 = f1_score(y_train, y_train_pred, average=average_method, pos_label=class_1)                       # F1-Score
train_accuracy = accuracy_score(y_train, y_train_pred)                                                      # Training accuracy
# Test metrics
test_precision = precision_score(y_test, y_test_pred, average=average_method, pos_label=class_1)            # Precision
test_recall = recall_score(y_test, y_test_pred, average=average_method, pos_label=class_1)                  # Recall
test_f1 = f1_score(y_test, y_test_pred, average=average_method, pos_label=class_1)                          # F1-Score  
test_accuracy = accuracy_score(y_test, y_test_pred)                                                         # Test accuracy 
print("\nPerformance Metrics Comparison:")
print("Metric\t\tTraining\tTest\t\tDifference")
print("-" * 55)
print(f"Accuracy\t{train_accuracy:.4f}\t\t{test_accuracy:.4f}\t\t{train_accuracy - test_accuracy:.4f}")         
print(f"Precision\t{train_precision:.4f}\t\t{test_precision:.4f}\t\t{train_precision - test_precision:.4f}")
print(f"Recall\t\t{train_recall:.4f}\t\t{test_recall:.4f}\t\t{train_recall - test_recall:.4f}")
print(f"F1-Score\t{train_f1:.4f}\t\t{test_f1:.4f}\t\t{train_f1 - test_f1:.4f}")
# Visualize confusion matrices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))                               # Create subplots for confusion matrices
# Training confusion matrix             
im1 = ax1.imshow(train_cm, interpolation='nearest', cmap=plt.cm.Blues)              # Display training confusion matrix
ax1.set_title('Training Set Confusion Matrix')                                      # Set title for training confusion matrix
ax1.set_ylabel('True Label')                                                        # Set y-label for training confusion matrix
ax1.set_xlabel('Predicted Label')                                                   # Set x-label for training confusion matrix
tick_marks = np.arange(len(best_knn.classes_))                                      # Create tick marks for classes
ax1.set_xticks(tick_marks)                                                          # Set x-ticks for training confusion matrix
ax1.set_yticks(tick_marks)                                                          # Set y-ticks for training confusion matrix
ax1.set_xticklabels(best_knn.classes_)                                              # Set x-tick labels to class names
ax1.set_yticklabels(best_knn.classes_)                                              # Set y-tick labels to class names
# Add text annotations
for i in range(train_cm.shape[0]):                                                  # Loop through each row of the confusion matrix
    for j in range(train_cm.shape[1]):                                              # Loop through each column of the confusion matrix
        ax1.text(j, i, train_cm[i, j], ha="center", va="center", color="white" if train_cm[i, j] > train_cm.max()/2 else "black")   # Add text annotation for each cell
# Test confusion matrix
im2 = ax2.imshow(test_cm, interpolation='nearest', cmap=plt.cm.Blues)               # Display test confusion matrix
ax2.set_title('Test Set Confusion Matrix')                                          # Set title for test confusion matrix 
ax2.set_ylabel('True Label')                                                        # Set y-label for test confusion matrix
ax2.set_xlabel('Predicted Label')                                                   # Set x-label for test confusion matrix
ax2.set_xticks(tick_marks)                                                          # Set x-ticks for test confusion matrix
ax2.set_yticks(tick_marks)                                                          # Set y-ticks for test confusion matrix
ax2.set_xticklabels(best_knn.classes_)                                              # Set x-tick labels to class names
ax2.set_yticklabels(best_knn.classes_)                                              # Set y-tick labels to class names
# Add text annotations
for i in range(test_cm.shape[0]):                                                   # Loop through each row of the confusion matrix
    for j in range(test_cm.shape[1]):                                               # Loop through each column of the confusion matrix
        ax2.text(j, i, test_cm[i, j], ha="center", va="center", color="white" if test_cm[i, j] > test_cm.max()/2 else "black")      # Add text annotation for each cell
plt.tight_layout()
plt.show()

print("\nMODEL LEARNING OUTCOME ANALYSIS:")
accuracy_diff = train_accuracy - test_accuracy                                      # Calculate differences in performance metrics
precision_diff = train_precision - test_precision                                   # Precision difference
recall_diff = train_recall - test_recall                                            # Recall difference                         
f1_diff = train_f1 - test_f1                                                        # F1-Score difference
avg_diff = np.mean([accuracy_diff, precision_diff, recall_diff, f1_diff])           # Calculate average difference across metrics
print(f"Average performance difference (Train - Test): {avg_diff:.4f}")                 
if avg_diff > 0.15:                                                                 #    
    print("INFERENCE: OVERFITTING")
elif avg_diff < -0.05:
    print("INFERENCE: UNDERFITTING")
elif abs(avg_diff) <= 0.05:
    print("INFERENCE: GOOD FIT")
else:
    print("INFERENCE: MODERATE OVERFITTING")

print("----------------------------------------------------------------------------------------------------------")