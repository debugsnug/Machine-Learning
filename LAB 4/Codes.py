
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Load the main dataset for all questions except A2
file_path = r"D:\\OneDrive - Amrita vishwa vidyapeetham\\SEM 5\\23CSE301 Machine Learning\\LAB\\Machine-Learning\\LAB 3\\20231225_dfall_obs_data_and_spectral_features_revision1_n469.csv"
data = pd.read_csv(file_path)

print("----------------------------------------------------------------------------------------------------------")

# A1. Classification evaluation
data["class"] = data["CallerSex"].astype('category').cat.codes          # Convert 'CallerSex' into categorical codes for classification
X = data[['F1','F2','F3','F4']].values                                # Features (input variables)
y = data['class'].values                                              # Target variable (the class labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)  # Split data into train and test sets (70-30 split)
knn = KNeighborsClassifier(n_neighbors=5)                 # Initialize K-Nearest Neighbors classifier with 5 neighbors
knn.fit(X_train, y_train)                                 # Train the model on the training data
y_train_pred = knn.predict(X_train)                       # Predict class labels for the training set
y_test_pred = knn.predict(X_test)                         # Predict class labels for the test set
print("A1 Answers: ")
print("Training Confusion Matrix:") 
print(confusion_matrix(y_train, y_train_pred)) 
print("\nTest Confusion Matrix:") 
print(confusion_matrix(y_test, y_test_pred)) 
print("\nTraining Classification Report:") 
print(classification_report(y_train, y_train_pred))
print("\nTest Classification Report:")
print(classification_report(y_test, y_test_pred))           # Show detailed metrics for test data
train_acc = (y_train == y_train_pred).mean()                # Calculate training accuracy (correct predictions / total predictions)
test_acc = (y_test == y_test_pred).mean()                   # Calculate test accuracy
print(f"\nTraining Accuracy: {train_acc:.2f}")
print(f"Test Accuracy: {test_acc:.2f}")
if train_acc > 0.90 and (train_acc - test_acc) > 0.15:  # Check if model is overfitting
    print("\nObservation: Model is Overfitting.")
elif train_acc < 0.70 and test_acc < 0.70:              # Check if model is underfitting
    print("\nObservation: Model is Underfitting.")
else:                                                   # Model is performing reasonably well
    print("\nObservation: Model is Regular Fit (good generalization).")

print("----------------------------------------------------------------------------------------------------------")

# A2. Regression metrics
file_path = r"D:\\OneDrive - Amrita vishwa vidyapeetham\\SEM 5\\23CSE301 Machine Learning\\LAB\\Machine-Learning\\LAB 2\\Lab Session Data.xlsx"  
df = pd.read_excel(file_path, sheet_name="IRCTC Stock Price")                                      # Read the specific sheet from the Excel file
df = df.dropna(subset=["Open", "High", "Low", "Price"])                                            # Drop rows with missing values in the relevant columns
X = df[["Open", "High", "Low"]].values                                      # Features: Open, High, Low prices
y = df["Price"].values                                                      # Target variable: Price
from sklearn.linear_model import LinearRegression
reg = LinearRegression()                                                    # Initialize linear regression model
reg.fit(X, y)                                                               # Fit the model to the data
y_pred = reg.predict(X)                                                     # Predict the target variable using the model
print("A2 Answers: ")
print("\nSample actual vs predicted:")
print(pd.DataFrame({"Actual": y[:5], "Predicted": y_pred[:5]}))
mse = np.mean((y - y_pred) ** 2)                                       # Mean Squared Error        
rmse = np.sqrt(mse)                                                         # Root Mean Squared Error     
mape = np.mean(np.abs((y - y_pred) / (y + 1e-8))) * 100           # Mean Absolute Percentage Error
ss_total = np.sum((y - np.mean(y)) ** 2)                          # Total Sum of Squares
ss_residual = np.sum((y - y_pred) ** 2)                                # Residual Sum of Squares
r2 = 1 - (ss_residual / ss_total)                                           # R² score      
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.4f}%")
print(f"R²: {r2:.4f}")

print("----------------------------------------------------------------------------------------------------------")

# A3. Generate training data
def generate_training_data():
    np.random.seed(42)                                      # Set the random seed for reproducibility
    X = np.random.uniform(1, 10, 20)                        # Generate 20 random points for X feature  
    Y = np.random.uniform(1, 10, 20)                        # Generate 20 random points for Y feature
    classes = []                                            # Initialize an empty list for class labels 
    for i in range(20):                                     # Iterate through each point
        if Y[i] > X[i]:                                     # If Y is greater than X, assign class 1 (Red)
            classes.append(1)                               
        else:                                               # If Y is less than or equal to X, assign class 0 (Blue)
            classes.append(0)                               # Append class label to the list
    df = pd.DataFrame({
        'X': X,                                             # Create a DataFrame with X feature
        'Y': Y,                                             # Create a DataFrame with Y feature
        'Class': classes                                    # Create a DataFrame with Class labels
    })
    print("\nA3 Answers: ")
    print("Generated Training Data:")
    print(df)                                               # Display the generated training data
    plt.figure(figsize=(10, 8))                             # Create a scatter plot of the training data    
    class0 = df[df['Class'] == 0]                           # Filter points belonging to class 0 (Blue)
    class1 = df[df['Class'] == 1]                           # Filter points belonging to class 1 (Red)
    plt.scatter(class0['X'], class0['Y'], c='blue', label='Class 0 (Blue)', s=100, alpha=0.7, edgecolors='k')           # Scatter plot for class 0
    plt.scatter(class1['X'], class1['Y'], c='red', label='Class 1 (Red)', s=100, alpha=0.7, edgecolors='k')             # Scatter plot for class 1
    plt.plot([1, 10], [1, 10], 'k--', alpha=0.5, label='Decision Boundary (y = x)')   
    plt.xlabel('X Feature', fontsize=12)                                        # Set the x-axis label
    plt.ylabel('Y Feature', fontsize=12)                                        # Set the y-axis label
    plt.title('Training Data with Class Labels', fontsize=14)                   # Set the title of the plot
    plt.grid(True, alpha=0.3)                                           # Add grid lines to the plot
    plt.legend()                                                        # Add a legend to the plot
    plt.xlim(0, 11)                                                     # Set the x-axis limits
    plt.ylim(0, 11)                                                     # Set the y-axis limits
    plt.tight_layout()                                                  # Adjust layout to prevent overlap
    plt.show()
    class_counts = df['Class'].value_counts().sort_index()              # Count the number of points in each class
    print(f"\nClass Distribution:")                                      
    print(f"Class 0 (Blue): {class_counts[0]} points")                  # Print the count of class 0 points
    print(f"Class 1 (Red): {class_counts[1]} points")                   # Print the count of class 1 points
    return df                                                           # Return the generated training data DataFrame

train_data = generate_training_data()

print("----------------------------------------------------------------------------------------------------------") 

# A4. Train a model using the generated data
def generate_test_data():
    x = np.arange(0, 10.1, 0.1)  # Generate test data points for X feature
    y = np.arange(0, 10.1, 0.1)  # Generate test data points for Y feature
    X, Y = np.meshgrid(x, y)     # Create a meshgrid for X and Y features
    X_flat = X.flatten()         # Flatten the X feature grid to a 1D array
    Y_flat = Y.flatten()         # Flatten the Y feature grid to a 1D array
    test_df = pd.DataFrame({   
        'X': X_flat,             # Create a DataFrame for test data with X feature
        'Y': Y_flat              # Create a DataFrame for test data with Y feature
    })  
    return test_df, X, Y        # Generate the test data DataFrame and return it along with the meshgrid

# Function for kNN classification and predicting class labels for the test data
def knn_classification(train_df, test_df, k=3): 
    X_train = train_df[['X', 'Y']].values       # Extract X and Y features from the training DataFrame
    y_train = train_df['Class'].values          # Extract class labels from the training DataFrame
    knn = KNeighborsClassifier(n_neighbors=k)   # Initialize kNN classifier with specified k value
    knn.fit(X_train, y_train)                   # Fit the kNN model on the training data
    X_test = test_df[['X', 'Y']].values         # Extract X and Y features from the test DataFrame
    y_pred = knn.predict(X_test)                # Predict class labels for the test data using the trained kNN model
    test_df['Predicted_Class'] = y_pred         # Add predicted class labels to the test DataFrame
    return test_df, knn

# Function to visualize the classification results of the kNN model
def visualize_classification_results(train_df, test_df, X_grid, Y_grid, k=3):
    plt.figure(figsize=(10, 6))  # Set the figure size for the plot
    # Plot the test points with predicted class colors
    plt.scatter(test_df[test_df['Predicted_Class'] == 0]['X'], test_df[test_df['Predicted_Class'] == 0]['Y'], c='blue', s=30, alpha=0.6, label='Predicted Class 0')
    plt.scatter(test_df[test_df['Predicted_Class'] == 1]['X'], test_df[test_df['Predicted_Class'] == 1]['Y'], c='red', s=30, alpha=0.6, label='Predicted Class 1')
    # Plot the training data points
    plt.scatter(train_df[train_df['Class'] == 0]['X'], train_df[train_df['Class'] == 0]['Y'], c='blue', edgecolors='k', s=150, marker='o', label='Training Class 0')
    plt.scatter(train_df[train_df['Class'] == 1]['X'], train_df[train_df['Class'] == 1]['Y'], c='red', edgecolors='k', s=150, marker='o', label='Training Class 1')
    # Customize the plot
    plt.xlabel('X Feature')
    plt.ylabel('Y Feature')
    plt.title(f'Test Data Classification (k={k})')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()          # Show the plot with classification results

test_data, X_grid, Y_grid = generate_test_data()
classified_test_data, knn_model = knn_classification(train_data, test_data, k=3)
visualize_classification_results(train_data, classified_test_data, X_grid, Y_grid, k=3)

# print("----------------------------------------------------------------------------------------------------------")

# A5. Compare kNN classification for multiple k values
def compare_multiple_k_values(train_df, test_df, X_grid, Y_grid, k_values=[1, 3, 5, 7, 9, 11, 15, 19]):                 
    predictions_dict = {}  # Dictionary to store predictions for each k
    X_train = train_df[['X', 'Y']].values  # Extract X and Y features from training set
    y_train = train_df['Class'].values  # Extract target variable (Class) from training set
    X_test = test_df[['X', 'Y']].values  # Extract X and Y features from test set
    
    # Create predictions for each k
    for k in k_values:  # Loop over each k value in the provided list
        knn = KNeighborsClassifier(n_neighbors=k)  # Initialize KNN classifier with current k
        knn.fit(X_train, y_train)  # Train the classifier on training data
        y_pred = knn.predict(X_test)  # Make predictions on the test data
        predictions_dict[k] = y_pred  # Store the predictions for the current k
    
    # Create subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # Create a 2x4 grid of subplots
    axes = axes.flatten()  # Flatten the axes array for easier indexing

    # Plot decision boundaries and training data
    for i, k in enumerate(k_values):  # Loop over each k value and subplot index
        if i < len(axes):  # Make sure we don't exceed the number of subplots
            predictions = predictions_dict[k]  # Get the predictions for current k
            Z = predictions.reshape(X_grid.shape)  # Reshape predictions to match grid shape
            # Plot decision boundaries
            axes[i].contourf(X_grid, Y_grid, Z, alpha=0.3, cmap=plt.cm.coolwarm)  # Plot contours as decision boundary
            # Plot training data points
            class0 = train_df[train_df['Class'] == 0]  # Select class 0 data points
            class1 = train_df[train_df['Class'] == 1]  # Select class 1 data points
            axes[i].scatter(class0['X'], class0['Y'], c='blue', edgecolors='k', s=100, marker='o', label='Class 0')  # Plot class 0 points
            axes[i].scatter(class1['X'], class1['Y'], c='red', edgecolors='k', s=100, marker='o', label='Class 1')  # Plot class 1 points
            # Diagonal line (optional, to visualize class separation)
            axes[i].plot([0, 10], [0, 10], 'k--', alpha=0.7)  # Plot diagonal line for reference
            # Set labels and title
            axes[i].set_xlabel('X Feature')  # Set X-axis label
            axes[i].set_ylabel('Y Feature')  # Set Y-axis label
            axes[i].set_title(f'k={k}')  # Set the title of the plot showing current k
            axes[i].grid(True, alpha=0.3)  # Add grid with low transparency
            axes[i].set_xlim(0, 10)  # Set the X-axis limits
            axes[i].set_ylim(0, 10)  # Set the Y-axis limits
            axes[i].legend()  # Display the legend

    # Adjust layout
    plt.tight_layout()  # Automatically adjust subplot parameters for neatness
    plt.suptitle('kNN Decision Boundaries for Different k Values', fontsize=16, y=1.02)  # Add a super title above all subplots
    plt.show()  # Show the plot
    print("A5 Answers: ")
    print("Classification Summary for Different k Values:")  # Print header for summary
    for k in k_values:  # Loop over each k value
        y_pred = predictions_dict[k]  # Get the predictions for current k
        class0_count = np.sum(y_pred == 0)  # Count how many class 0 predictions
        class1_count = np.sum(y_pred == 1)  # Count how many class 1 predictions
        print(f"k={k:2d}: Class 0: {class0_count:5d}, Class 1: {class1_count:5d}")  # Print the counts for class 0 and class 1 for the current k

compare_multiple_k_values(train_data, test_data, X_grid, Y_grid, k_values=[1, 3, 5, 7, 9, 11, 15, 19])

# print("----------------------------------------------------------------------------------------------------------")

# A6. Analyze elephant call data using kNN classification
# Generate 20 random training samples from the dataset
train_df = data[['F1', 'F2', 'CallerSex']].dropna().sample(n=20, random_state=42)
train_df['Class'] = train_df['CallerSex'].map({'F': 0, 'M': 1})  # Female=0, Male=1
# Scatter plot of training data
plt.figure(figsize=(8, 6))
plt.scatter(train_df[train_df['Class'] == 0]['F1'], train_df[train_df['Class'] == 0]['F2'], c='blue', label='Female (Class 0)', s=80, edgecolors='k')
plt.scatter(train_df[train_df['Class'] == 1]['F1'], train_df[train_df['Class'] == 1]['F2'], c='red', label='Male (Class 1)', s=80, edgecolors='k')
plt.xlabel('F1')
plt.ylabel('F2')
plt.title('A6: Training Data (F1 vs F2)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Generate test data grid
x = np.linspace(train_df['F1'].min(), train_df['F1'].max(), 100)  # Adjust range based on training data
y = np.linspace(train_df['F2'].min(), train_df['F2'].max(), 100) 
X_grid, Y_grid = np.meshgrid(x, y)
test_df = pd.DataFrame({'F1': X_grid.flatten(), 'F2': Y_grid.flatten()})

# kNN classification for k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_df[['F1', 'F2']], train_df['Class'])
test_df['Predicted_Class'] = knn.predict(test_df[['F1', 'F2']])

# Scatter plot of test data predictions
plt.figure(figsize=(8, 6))
plt.scatter(test_df[test_df['Predicted_Class'] == 0]['F1'], test_df[test_df['Predicted_Class'] == 0]['F2'], c='blue', s=10, alpha=0.5, label='Predicted Class 0')
plt.scatter(test_df[test_df['Predicted_Class'] == 1]['F1'], test_df[test_df['Predicted_Class'] == 1]['F2'], c='red', s=10, alpha=0.5, label='Predicted Class 1')
plt.xlabel('F1')
plt.ylabel('F2')
plt.title('A6: Test Data Classification (k=3)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Repeat for multiple k values and plot decision boundaries
k_values = [1, 3, 5, 7, 9, 11, 15, 19]
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
for i, k in enumerate(k_values):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_df[['F1', 'F2']], train_df['Class'])
    Z = knn.predict(test_df[['F1', 'F2']]).reshape(X_grid.shape)
    axes[i].contourf(X_grid, Y_grid, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    axes[i].scatter(train_df[train_df['Class'] == 0]['F1'], train_df[train_df['Class'] == 0]['F2'], c='blue', edgecolors='k', s=80, label='Class 0')
    axes[i].scatter(train_df[train_df['Class'] == 1]['F1'], train_df[train_df['Class'] == 1]['F2'], c='red', edgecolors='k', s=80, label='Class 1')
    axes[i].set_title(f'k={k}')
    axes[i].set_xlabel('F1')
    axes[i].set_ylabel('F2')
    axes[i].set_xlim(0, 10)
    axes[i].set_ylim(0, 10)
    axes[i].grid(True, alpha=0.3)
    axes[i].legend()
plt.tight_layout()
plt.suptitle('A6: kNN Decision Boundaries for Different k Values', fontsize=16, y=1.02)
plt.show()

print("----------------------------------------------------------------------------------------------------------")

# A7. Hyperparameter tuning using GridSearchCV and RandomizedSearchCV

# Initialize the classifier
knn = KNeighborsClassifier()
# Corrected GridSearchCV
param_grid = {'n_neighbors': np.arange(1, 21)}  # Grid search will try each value of n_neighbors from 1 to 20
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy')
# Fit grid search
grid_search.fit(X_train, y_train)
# Output best parameters and score from grid search
print("A7 Answers: ")
print("Best k value (from GridSearchCV):", grid_search.best_params_['n_neighbors'])
print("Best accuracy (from GridSearchCV):", grid_search.best_score_)
# Corrected RandomizedSearchCV
param_dist = {'n_neighbors': np.arange(1, 21)}  # Random search will sample 10 random values from 1 to 20
random_search = RandomizedSearchCV(estimator=knn, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42)
# Fit random search
random_search.fit(X_train, y_train)
# Output best parameters and score from random search
print("Best k value (from RandomizedSearchCV):", random_search.best_params_['n_neighbors'])
print("Best accuracy (from RandomizedSearchCV):", random_search.best_score_)

print("----------------------------------------------------------------------------------------------------------")