
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
def create_call_type_classes(df):
    le = LabelEncoder()                                                     # Encode call types as classes
    df['class'] = le.fit_transform(df['Call_Type'])                         # Encode the 'Call_Type' column into numerical classes
    class_mapping = dict(zip(le.classes_, range(len(le.classes_))))         # Create a mapping of class names to numerical values
    print("Class mapping:", class_mapping)    
    return df, le                                                           # Return the modified DataFrame and the label encoder

def evaluate_classification_model(data):
    print("A1 Answers: ")
    data_classified, label_encoder = create_call_type_classes(data)                             # Create classes for call types
    feature_columns = ['F1', 'F2', 'F3', 'F4']                                                  # Formant frequencies                                   
    available_features = [col for col in feature_columns if col in data.columns]                # Check if feature columns are present in the data
    if len(available_features) < 2:                                                             # Warning: Not enough acoustic features found. Using available numerical columns.
        print("Warning: Not enough acoustic features found. Using available numerical columns.")
        numerical_cols = data.select_dtypes(include=[np.number]).columns                                # If not enough acoustic features, use numerical columns
        available_features = [col for col in numerical_cols if col not in ['class', 'selec']][:4]       # Select first 4 numerical columns excluding 'class' and 'selec'
    print(f"Using features: {available_features}")
    X = data_classified[available_features].fillna(0)                                                           # Fill NaN with 0
    y = data_classified['class']                                                                                # Split the data into features and target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)       # Split the data into training and test sets
    scaler = StandardScaler()                                    # Standardize the features
    X_train_scaled = scaler.fit_transform(X_train)               # Fit the scaler on training data and transform it
    X_test_scaled = scaler.transform(X_test)                     # Transform the test data using the fitted scaler
    knn = KNeighborsClassifier(n_neighbors=5)                    # Initialize kNN classifier with k=5
    knn.fit(X_train_scaled, y_train)                             # Fit the kNN model on the training data
    y_train_pred = knn.predict(X_train_scaled)                   # Predict on training data
    y_test_pred = knn.predict(X_test_scaled)                     # Predict on test data
    train_accuracy = accuracy_score(y_train, y_train_pred)       # Calculate accuracy on training data
    test_accuracy = accuracy_score(y_test, y_test_pred)          # Calculate accuracy on test data
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    train_cm = confusion_matrix(y_train, y_train_pred)              # Confusion matrix for training data
    test_cm = confusion_matrix(y_test, y_test_pred)                 # Confusion matrix for test data
    print("\nTraining Confusion Matrix:")
    print(train_cm)                                                 # Print training confusion matrix
    print("\nTest Confusion Matrix:")
    print(test_cm)                                                  # Print test confusion matrix
    print("\nTraining Classification Report:")
    print(classification_report(y_train, y_train_pred))             # Print classification report for training data
    print("\nTest Classification Report:")
    print(classification_report(y_test, y_test_pred))               # Print classification report for test data
    if train_accuracy > 0.95 and test_accuracy < 0.8:               # If training accuracy is high and test accuracy is low, model is overfitting
        print("Model appears to be OVERFITTING")
    elif train_accuracy < 0.7 and test_accuracy < 0.7:              # If both training and test accuracy are low, model is underfitting
        print("Model appears to be UNDERFITTING")
    else:                                                           # If training accuracy is moderate and test accuracy is acceptable, model is well-fitted    
        print("Model appears to have REGULAR FIT")   
    return knn, scaler, X_train_scaled, X_test_scaled, y_train, y_test

#print("----------------------------------------------------------------------------------------------------------")

# A2. Regression metrics
def calculate_regression_metrics(y_true, y_pred):
    print("A2 Answers: ")
    mse = np.mean((y_true - y_pred) ** 2)                                       # Mean Squared Error        
    rmse = np.sqrt(mse)                                                         # Root Mean Squared Error     
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100           # Mean Absolute Percentage Error
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)                          # Total Sum of Squares
    ss_residual = np.sum((y_true - y_pred) ** 2)                                # Residual Sum of Squares
    r2 = 1 - (ss_residual / ss_total)                                           # R² score      
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4f}%")
    print(f"R²: {r2:.4f}")

def demonstrate_regression_metrics_lab2():
    import os
    excel_path = os.path.join(os.path.dirname(__file__), "..", "LAB 2", "Lab Session Data.xlsx")        # Path to the Excel file
    df = pd.read_excel(excel_path, sheet_name="IRCTC Stock Price")                                      # Read the specific sheet from the Excel file
    df = df.dropna(subset=["Open", "High", "Low", "Price"])                                             # Drop rows with missing values in the relevant columns
    X = df[["Open", "High", "Low"]].values                                      # Features: Open, High, Low prices
    y = df["Price"].values                                                      # Target variable: Price
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()                                                    # Initialize linear regression model
    reg.fit(X, y)                                                               # Fit the model to the data
    y_pred = reg.predict(X)                                                     # Predict the target variable using the model
    calculate_regression_metrics(y, y_pred)                                     # Calculate regression metrics
    print("\nSample actual vs predicted:")
    print(pd.DataFrame({"Actual": y[:5], "Predicted": y_pred[:5]}))

#print("----------------------------------------------------------------------------------------------------------")

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

# print("----------------------------------------------------------------------------------------------------------") 

# A4. Train a model using the generated data
def generate_test_data():
    x = np.arange(0, 10.1, 0.1)                             # Generate test data points for X feature
    y = np.arange(0, 10.1, 0.1)                             # Generate test data points for Y feature
    X, Y = np.meshgrid(x, y)                                # Create a meshgrid for X and Y features
    X_flat = X.flatten()                                    # Flatten the X feature grid to a 1D array
    Y_flat = Y.flatten()                                    # Flatten the Y feature grid to a 1D array
    test_df = pd.DataFrame({   
        'X': X_flat,                                        # Create a DataFrame for test data with X feature
        'Y': Y_flat                                         # Create a DataFrame for test data with Y feature
    })  
    return test_df, X, Y                                    # Generate the test data DataFrame and return it along with the meshgrid

def knn_classification(train_df, test_df, k=3):             # Train a kNN classifier using the generated training data
    X_train = train_df[['X', 'Y']].values                   # Extract X and Y features from the training DataFrame
    y_train = train_df['Class'].values                      # Extract class labels from the training DataFrame
    knn = KNeighborsClassifier(n_neighbors=k)               # Initialize kNN classifier with specified k value
    knn.fit(X_train, y_train)                               # Fit the kNN model on the training data
    X_test = test_df[['X', 'Y']].values                     # Extract X and Y features from the test DataFrame
    y_pred = knn.predict(X_test)                            # Predict class labels for the test data using the trained kNN model
    test_df['Predicted_Class'] = y_pred
    print("A4 Answers: ")                         # Add predicted class labels to the test DataFrame
    print(f"Test Data Summary:")            
    print(f"Total points: {len(test_df)}")
    print(f"Predicted Class 0 (Blue): {len(test_df[test_df['Predicted_Class'] == 0])}")
    print(f"Predicted Class 1 (Red): {len(test_df[test_df['Predicted_Class'] == 1])}")
    return test_df, knn

def visualize_classification_results(train_df, test_df, X_grid, Y_grid, k=3):                      # Visualize the classification results of the kNN model
    plt.figure(figsize=(15, 6))                                                                    # Set the figure size for the plot

    plt.subplot(1, 2, 1)                                                                           # Plot 1: Decision boundary of the kNN classifier
    predictions = test_df['Predicted_Class'].values                                                # Create a contour plot for the decision boundary
    Z = predictions.reshape(X_grid.shape)                                                          # Reshape the predicted class labels to match the grid shape
    plt.contourf(X_grid, Y_grid, Z, alpha=0.3, cmap=plt.cm.coolwarm)                               # Create a filled contour plot for the decision boundary
    class0 = train_df[train_df['Class'] == 0]                                                      # Filter points belonging to class 0 (Blue)
    class1 = train_df[train_df['Class'] == 1]                                                      # Filter points belonging to class 1 (Red)
    plt.scatter(class0['X'], class0['Y'], c='blue', edgecolors='k', s=150, marker='o', label='Training Class 0')        # Scatter plot for class 0
    plt.scatter(class1['X'], class1['Y'], c='red', edgecolors='k', s=150, marker='o', label='Training Class 1')         # Scatter plot for class 1
    plt.plot([0, 10], [0, 10], 'k--', alpha=0.7, label='True boundary (y = x)')                     # Plot the true decision boundary
    plt.xlabel('X Feature')                         
    plt.ylabel('Y Feature')
    plt.title(f'kNN Decision Boundary (k={k})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    
    plt.subplot(1, 2, 2)                                                                            # Plot 2: Sample of test points
    sample_size = 1000                                                                              # Sample size for test points
    test_sample = test_df.sample(sample_size, random_state=42)                                      # Randomly sample test points for visualization
    test_class0 = test_sample[test_sample['Predicted_Class'] == 0]                                  # Filter points predicted as class 0 (Blue)
    test_class1 = test_sample[test_sample['Predicted_Class'] == 1]                                  # Filter points predicted as class 1 (Red)
    plt.scatter(test_class0['X'], test_class0['Y'], c='blue', s=30, alpha=0.6, label='Test Class 0')                    # Scatter plot for class 0  
    plt.scatter(test_class1['X'], test_class1['Y'], c='red', s=30, alpha=0.6, label='Test Class 1')                     # Scatter plot for class 1
    plt.scatter(class0['X'], class0['Y'], c='blue', edgecolors='k', s=150, marker='o', label='Training Class 0')        # Training points for class 0
    plt.scatter(class1['X'], class1['Y'], c='red', edgecolors='k', s=150, marker='o', label='Training Class 1')         # Training points for class 1   
    plt.plot([0, 10], [0, 10], 'k--', alpha=0.7)                                                    # Plot the true decision boundary
    plt.xlabel('X Feature')                                                                         
    plt.ylabel('Y Feature')
    plt.title(f'Test Data Classification (k={k})')
    plt.grid(True, alpha=0.3)                                                                       
    plt.legend()
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    
    plt.tight_layout()                                  # Adjust layout to prevent overlap
    plt.show()                                          # Show the plot with classification results

# print("----------------------------------------------------------------------------------------------------------")

# A5. Compare kNN classification for multiple k values
def compare_multiple_k_values(train_df, test_df, X_grid, Y_grid, k_values=[1, 3, 5, 7, 9, 11, 15, 19]):                 
    predictions_dict = {}
    X_train = train_df[['X', 'Y']].values
    y_train = train_df['Class'].values
    X_test = test_df[['X', 'Y']].values
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        predictions_dict[k] = y_pred

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    for i, k in enumerate(k_values):
        if i < len(axes):
            predictions = predictions_dict[k]
            Z = predictions.reshape(X_grid.shape)
            axes[i].contourf(X_grid, Y_grid, Z, alpha=0.3, cmap=plt.cm.coolwarm)
            class0 = train_df[train_df['Class'] == 0]
            class1 = train_df[train_df['Class'] == 1]
            axes[i].scatter(class0['X'], class0['Y'], c='blue', edgecolors='k', s=100, marker='o')
            axes[i].scatter(class1['X'], class1['Y'], c='red', edgecolors='k', s=100, marker='o')
            axes[i].plot([0, 10], [0, 10], 'k--', alpha=0.7)
            axes[i].set_xlabel('X Feature')
            axes[i].set_ylabel('Y Feature')
            axes[i].set_title(f'k={k}')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(0, 10)
            axes[i].set_ylim(0, 10)
    plt.tight_layout()
    plt.suptitle('kNN Decision Boundaries for Different k Values', fontsize=16, y=1.02)
    plt.show()
    print("A5 Answers: ")
    print("Classification Summary for Different k Values:")
    for k in k_values:
        y_pred = predictions_dict[k]
        class0_count = np.sum(y_pred == 0)
        class1_count = np.sum(y_pred == 1)
        print(f"k={k:2d}: Class 0: {class0_count:5d}, Class 1: {class1_count:5d}")

# print("----------------------------------------------------------------------------------------------------------")

# A6. Analyze elephant call data using kNN classification
def analyze_elephant_call_data(data):
    if 'Context' in data.columns:                                                               # Check if 'Context' column exists in the dataset
        context_counts = data['Context'].value_counts()                                         # Count occurrences of each context 
        top_contexts = context_counts.head(2).index.tolist()                                    # Get the top 2 contexts based on their counts
        print("A6 Answers: ")
        print(f"Using top 2 contexts: {top_contexts}")                                          
        filtered_data = data[data['Context'].isin(top_contexts)].copy()                             # Filter the data to include only the top 2 contexts
        filtered_data['binary_class'] = (filtered_data['Context'] == top_contexts[0]).astype(int)       # Create a binary class column based on the top context
        feature_cols = ['F1', 'F2']                                                                     # Formant frequencies
        if not all(col in filtered_data.columns for col in feature_cols):                               # If feature columns are not present, use numerical columns
            numerical_cols = filtered_data.select_dtypes(include=[np.number]).columns                       # Select numerical columns
            feature_cols = [col for col in numerical_cols if col not in ['binary_class', 'selec']][:2]      # Select the first 2 numerical columns
        print(f"Using features: {feature_cols}")    
        clean_data = filtered_data[feature_cols + ['binary_class']].dropna()                    # Clean the data by selecting feature columns and dropping NaN values
        if len(clean_data) > 20:                                                                # Check if there are enough clean data points for analysis
            X = clean_data[feature_cols].values                                                 # Extract feature values
            y = clean_data['binary_class'].values                                                           # Extract binary class labels
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)       # Split the data into training and test sets
            scaler = StandardScaler()                                                           # Standardize the features
            X_train_scaled = scaler.fit_transform(X_train)                                      # Fit the scaler on training data and transform it
            X_test_scaled = scaler.transform(X_test)                                            # Transform the test data using the fitted scaler
            k_values = [1, 3, 5, 7, 9]                                                          # Define a list of k values to compare  
            plt.figure(figsize=(15, 10))                                                        # Create a figure for plotting decision boundaries
            for i, k in enumerate(k_values):                                                    # Iterate through each k value
                knn = KNeighborsClassifier(n_neighbors=k)                                       # Initialize kNN classifier with the current k value
                knn.fit(X_train_scaled, y_train)                                                    # Fit the kNN model on the training data
                h = 0.02                                                                            # Step size for the mesh grid
                x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1           # Define the range for the x-axis
                y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1           # Define the range for the y-axis
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))            # Create a mesh grid for plotting decision boundaries
                Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])                                          # Predict class labels for each point in the mesh grid
                Z = Z.reshape(xx.shape)                                                             # Reshape the predicted class labels to match the grid shape
                plt.subplot(2, 3, i+1)                                                              # Create a subplot for the current k value
                plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)                            # Create a filled contour plot for the decision boundary
                
                # Plot training points
                class0_mask = y_train == 0                                                          # Mask for class 0 (Blue)
                class1_mask = y_train == 1                                                          # Mask for class 1 (Red)                                     
                plt.scatter(X_train_scaled[class0_mask, 0], X_train_scaled[class0_mask, 1], c='blue', marker='o', s=50, label=f'{top_contexts[1]}')    # Scatter plot for class 0
                plt.scatter(X_train_scaled[class1_mask, 0], X_train_scaled[class1_mask, 1], c='red', marker='s', s=50, label=f'{top_contexts[0]}')     # Scatter plot for class 1
                plt.xlabel(f'{feature_cols[0]} (scaled)')                                       
                plt.ylabel(f'{feature_cols[1]} (scaled)')
                plt.title(f'Elephant Calls: k={k}')
                plt.legend()
                plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            print(f"\nElephant Call Classification Results:")
            for k in k_values:                                                              # Iterate through each k value
                knn = KNeighborsClassifier(n_neighbors=k)                                   # Initialize kNN classifier with the current k value
                knn.fit(X_train_scaled, y_train)                                            # Fit the kNN model on the training data
                y_pred = knn.predict(X_test_scaled)                                         # Predict class labels for the test data
                accuracy = accuracy_score(y_test, y_pred)                                   # Calculate accuracy
                print(f"k={k}: Accuracy = {accuracy:.4f}")        
        else:
            print("Not enough clean data for elephant call analysis")                       
    else:
        print("Context column not found in the dataset")

# print("----------------------------------------------------------------------------------------------------------")

# A7. Hyperparameter tuning using GridSearchCV and RandomizedSearchCV
def hyperparameter_tuning(data):                                                            # Check if 'Context' column exists in the dataset
    if 'Context' in data.columns:                                                           # If 'Context' column exists, proceed with hyperparameter tuning
        context_counts = data['Context'].value_counts()                                     # Count occurrences of each context
        top_contexts = context_counts.head(2).index.tolist()                                # Get the top 2 contexts based on their counts
        filtered_data = data[data['Context'].isin(top_contexts)].copy()                                 # Filter the data to include only the top 2 contexts
        filtered_data['binary_class'] = (filtered_data['Context'] == top_contexts[0]).astype(int)       # Create a binary class column based on the top context
        feature_cols = ['F1', 'F2']                                                             # Formant frequencies
        if not all(col in filtered_data.columns for col in feature_cols):                       # If feature columns are not present, use numerical columns
            numerical_cols = filtered_data.select_dtypes(include=[np.number]).columns                        # Select numerical columns
            feature_cols = [col for col in numerical_cols if col not in ['binary_class', 'selec']][:2]       # Select the first 2 numerical columns
        clean_data = filtered_data[feature_cols + ['binary_class']].dropna()
        if len(clean_data) > 20:                                                # Check if there are enough clean data points for hyperparameter tuning     
            X = clean_data[feature_cols].values                                 # Feature matrix
            y = clean_data['binary_class'].values                               # Target vector
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)       # Split the data into training and test sets
            scaler = StandardScaler()                                           # Initialize a StandardScaler object
            X_train_scaled = scaler.fit_transform(X_train)                      # Fit the scaler on training data and transform it
            X_test_scaled = scaler.transform(X_test)                            # Transform the test data using the fitted scaler   

            print("A7 Answers: ")
            print("Performing GridSearchCV...")
            param_grid = {'n_neighbors': list(range(1, 21))}                # Define the parameter grid for kNN hyperparameter tuning    
            knn_grid = KNeighborsClassifier()                                                           # Initialize kNN classifier     
            grid_search = GridSearchCV(knn_grid, param_grid, cv=5, scoring='accuracy', n_jobs=-1)       # Perform grid search with cross-validation
            grid_search.fit(X_train_scaled, y_train)                                                    # Fit the grid search on the training data
            print(f"Best k (GridSearchCV): {grid_search.best_params_['n_neighbors']}")                
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            
            print("\nPerforming RandomizedSearchCV...")
            param_dist = {'n_neighbors': list(range(1, 51))}               # Define the parameter distribution for kNN hyperparameter tuning
            random_search = RandomizedSearchCV(knn_grid, param_dist, n_iter=20, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)       # Perform randomized search with cross-validation
            random_search.fit(X_train_scaled, y_train)                                                                                      # Fit the randomized search on the training data
            print(f"Best k (RandomizedSearchCV): {random_search.best_params_['n_neighbors']}")                                              
            print(f"Best cross-validation score: {random_search.best_score_:.4f}")
            
            # Plot validation curve
            from sklearn.model_selection import validation_curve
            k_range = list(range(1, 21))                                        # Range of k values to evaluate
            train_scores, val_scores = validation_curve(KNeighborsClassifier(), X_train_scaled, y_train, param_name='n_neighbors', param_range=k_range, cv=5, scoring='accuracy', n_jobs=-1)    # Evaluate the model on the training data with different k values
            train_mean = np.mean(train_scores, axis=1)                          # Calculate the mean of the training scores
            train_std = np.std(train_scores, axis=1)                            # Calculate the standard deviation of the training scores
            val_mean = np.mean(val_scores, axis=1)                              # Calculate the mean of the validation scores
            val_std = np.std(val_scores, axis=1)                                # Calculate the standard deviation of the validation scores
            plt.figure(figsize=(10, 6))                                                             # Create a figure for the validation curve plot
            plt.plot(k_range, train_mean, 'o-', color='blue', label='Training accuracy')                                # Plot the training accuracy curve
            plt.fill_between(k_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')          # Fill the area under the training accuracy curve with a transparent blue color
            plt.plot(k_range, val_mean, 'o-', color='red', label='Validation accuracy')                                 # Plot the validation accuracy curve
            plt.fill_between(k_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')                   # Fill the area under the validation accuracy curve with a transparent red color
            plt.xlabel('k (Number of Neighbors)')
            plt.ylabel('Accuracy')
            plt.title('Validation Curve for kNN Classifier')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            
            # Test with best k
            best_k = grid_search.best_params_['n_neighbors']                            # Get the best k value from GridSearchCV
            best_knn = KNeighborsClassifier(n_neighbors=best_k)                         # Initialize kNN classifier with the best k value
            best_knn.fit(X_train_scaled, y_train)                                       # Fit the kNN model on the training data with the best k value  
            y_pred_best = best_knn.predict(X_test_scaled)                               # Predict class labels for the test data using the best kNN model
            print(f"\nTest accuracy with best k ({best_k}): {accuracy_score(y_test, y_pred_best):.4f}")
            
        else:
            print("Not enough clean data for hyperparameter tuning")
    else:
        print("Context column not found for hyperparameter tuning")
# print("----------------------------------------------------------------------------------------------------------")

def main():
    # A1. Classification evaluation
    model, scaler, X_train, X_test, y_train, y_test = evaluate_classification_model(data)

    print("-----------------------------------------------------------------------------------------------------------------------")

    # A2. Regression metrics demonstration (Lab 2 price prediction)
    demonstrate_regression_metrics_lab2()

    print("-----------------------------------------------------------------------------------------------------------------------")

    # A3. Generate training data
    train_data = generate_training_data()

    print("-----------------------------------------------------------------------------------------------------------------------")

    # A4. Test data classification
    test_data, X_grid, Y_grid = generate_test_data()
    classified_test_data, knn_model = knn_classification(train_data, test_data, k=3)
    visualize_classification_results(train_data, classified_test_data, X_grid, Y_grid, k=3)

    print("-----------------------------------------------------------------------------------------------------------------------")

    # A5. Multiple k values
    compare_multiple_k_values(train_data, test_data, X_grid, Y_grid)

    print("-----------------------------------------------------------------------------------------------------------------------")

    # A6. Elephant call data analysis
    analyze_elephant_call_data(data)

    print("-----------------------------------------------------------------------------------------------------------------------")

    # A7. Hyperparameter tuning
    hyperparameter_tuning(data)

if __name__ == "__main__":
    main()