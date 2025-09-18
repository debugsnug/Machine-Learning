import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# A1. Custom Functions for Neural Network Components

class ActivationFunctions:
    """Collection of activation functions"""
    
    @staticmethod
    def step(x):
        """Step activation function"""
        return np.where(x >= 0, 1, 0)
    
    @staticmethod
    def bipolar_step(x):
        """Bipolar step activation function"""
        return np.where(x >= 0, 1, -1)
    
    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function"""
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def tanh(x):
        """Hyperbolic tangent activation function"""
        return np.tanh(x)
    
    @staticmethod
    def relu(x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        """Leaky ReLU activation function"""
        return np.where(x > 0, x, alpha * x)

class NeuralNetworkComponents:
    """Neural network building blocks"""
    
    @staticmethod
    def summation_unit(inputs, weights, bias=None):
        """Summation unit: weighted sum of inputs"""
        if bias is not None:
            return np.dot(inputs, weights) + bias
        return np.dot(inputs, weights)
    
    @staticmethod
    def activation_unit(net_input, activation_func):
        """Apply activation function to net input"""
        return activation_func(net_input)
    
    @staticmethod
    def comparator_unit(target, output):
        """Calculate error between target and output"""
        return target - output

# A2. Custom Perceptron Implementation

class CustomPerceptron:
    def __init__(self, learning_rate=0.05, max_epochs=1000, tolerance=0.002, activation='step'):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.activation_name = activation
        self.weights = None
        self.bias = None
        self.errors = []
        self.epochs_to_converge = None
        
        # Set activation function
        if activation == 'step':
            self.activation_func = ActivationFunctions.step
        elif activation == 'bipolar_step':
            self.activation_func = ActivationFunctions.bipolar_step
        elif activation == 'sigmoid':
            self.activation_func = ActivationFunctions.sigmoid
        elif activation == 'tanh':
            self.activation_func = ActivationFunctions.tanh
        elif activation == 'relu':
            self.activation_func = ActivationFunctions.relu
        elif activation == 'leaky_relu':
            self.activation_func = ActivationFunctions.leaky_relu
    
    def fit(self, X, y, initial_weights=None):
        """Train the perceptron"""
        n_samples, n_features = X.shape
        
        # Initialize weights
        if initial_weights is not None:
            self.weights = np.array(initial_weights[1:])  # W1, W2, ...
            self.bias = initial_weights[0]  # W0
        else:
            self.weights = np.random.uniform(-0.5, 0.5, n_features)
            self.bias = np.random.uniform(-0.5, 0.5)
        
        # Training loop
        for epoch in range(self.max_epochs):
            total_error = 0
            
            for i in range(n_samples):
                # Forward pass
                net_input = NeuralNetworkComponents.summation_unit(X[i], self.weights, self.bias)
                prediction = NeuralNetworkComponents.activation_unit(net_input, self.activation_func)
                
                # Error calculation
                error = NeuralNetworkComponents.comparator_unit(y[i], prediction)
                total_error += error ** 2
                
                # Weight update
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error
            
            # Calculate mean squared error
            mse = total_error / n_samples
            self.errors.append(mse)
            
            # Check for convergence
            if mse <= self.tolerance:
                self.epochs_to_converge = epoch + 1
                print(f"Converged after {self.epochs_to_converge} epochs with {self.activation_name} activation")
                break
        
        if self.epochs_to_converge is None:
            self.epochs_to_converge = self.max_epochs
            print(f"Did not converge within {self.max_epochs} epochs with {self.activation_name} activation")
    
    def predict(self, X):
        """Make predictions"""
        net_input = NeuralNetworkComponents.summation_unit(X, self.weights, self.bias)
        return NeuralNetworkComponents.activation_unit(net_input, self.activation_func)
    
    def plot_error(self, title="Training Error"):
        """Plot training error vs epochs"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.errors) + 1), self.errors, 'b-', linewidth=2)
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error')
        plt.title(f'{title} - {self.activation_name.capitalize()} Activation')
        plt.grid(True, alpha=0.3)
        plt.show()

# Logic Gates Data
def get_and_gate_data():
    """AND gate truth table"""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    return X, y

def get_xor_gate_data():
    """XOR gate truth table"""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    return X, y

# A2. AND Gate with Step Activation Function
print("A2. AND Gate with Step Activation Function")
print("=" * 50)

X_and, y_and = get_and_gate_data()
initial_weights = [10, 0.2, -0.75]  # W0, W1, W2

perceptron_and_step = CustomPerceptron(learning_rate=0.05, activation='step')
perceptron_and_step.fit(X_and, y_and, initial_weights=initial_weights)
perceptron_and_step.plot_error("AND Gate - Step Activation")

# Test predictions
predictions = perceptron_and_step.predict(X_and)
print("AND Gate Predictions with Step Activation:")
for i in range(len(X_and)):
    print(f"Input: {X_and[i]}, Target: {y_and[i]}, Prediction: {predictions[i]}")

# A3. Compare different activation functions for AND gate
print("\nA3. AND Gate with Different Activation Functions")
print("=" * 50)

activation_functions = ['step', 'bipolar_step', 'sigmoid', 'relu']
convergence_results = {}

for activation in activation_functions:
    print(f"\nTraining with {activation} activation...")
    perceptron = CustomPerceptron(learning_rate=0.05, activation=activation)
    perceptron.fit(X_and, y_and, initial_weights=initial_weights)
    convergence_results[activation] = perceptron.epochs_to_converge

# Plot convergence comparison
plt.figure(figsize=(12, 6))
activations = list(convergence_results.keys())
epochs = list(convergence_results.values())
plt.bar(activations, epochs, color=['blue', 'green', 'red', 'orange'])
plt.xlabel('Activation Function')
plt.ylabel('Epochs to Convergence')
plt.title('AND Gate: Convergence Comparison Across Activation Functions')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nConvergence Results:")
for activation, epochs in convergence_results.items():
    print(f"{activation}: {epochs} epochs")

# A4. Vary learning rate for AND gate
print("\nA4. AND Gate with Varying Learning Rates")
print("=" * 50)

learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
lr_convergence = {}

for lr in learning_rates:
    perceptron = CustomPerceptron(learning_rate=lr, activation='step')
    perceptron.fit(X_and, y_and, initial_weights=initial_weights)
    lr_convergence[lr] = perceptron.epochs_to_converge

# Plot learning rate vs convergence
plt.figure(figsize=(12, 6))
lrs = list(lr_convergence.keys())
epochs_lr = list(lr_convergence.values())
plt.plot(lrs, epochs_lr, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Learning Rate')
plt.ylabel('Epochs to Convergence')
plt.title('AND Gate: Learning Rate vs Convergence Speed')
plt.grid(True, alpha=0.3)
plt.show()

print("\nLearning Rate Results:")
for lr, epochs in lr_convergence.items():
    print(f"Learning Rate {lr}: {epochs} epochs")

# A5. XOR Gate with different activation functions
print("\nA5. XOR Gate with Different Activation Functions")
print("=" * 50)

X_xor, y_xor = get_xor_gate_data()
xor_convergence = {}

for activation in activation_functions:
    print(f"\nTraining XOR with {activation} activation...")
    perceptron = CustomPerceptron(learning_rate=0.05, activation=activation, max_epochs=1000)
    perceptron.fit(X_xor, y_xor, initial_weights=initial_weights)
    xor_convergence[activation] = perceptron.epochs_to_converge
    
    # Test predictions
    predictions = perceptron.predict(X_xor)
    print(f"XOR Predictions with {activation}:")
    for i in range(len(X_xor)):
        print(f"Input: {X_xor[i]}, Target: {y_xor[i]}, Prediction: {predictions[i]}")

print("\nXOR Convergence Results:")
for activation, epochs in xor_convergence.items():
    print(f"{activation}: {epochs} epochs")

# A6. Customer Data Classification
print("\nA6. Customer Data Classification")
print("=" * 50)

# Customer data
customer_data = {
    'Customer': ['C_1', 'C_2', 'C_3', 'C_4', 'C_5', 'C_6', 'C_7', 'C_8', 'C_9', 'C_10'],
    'Candies': [20, 16, 27, 19, 24, 22, 15, 18, 21, 16],
    'Mangoes': [6, 3, 6, 1, 4, 1, 4, 4, 1, 2],
    'Milk_Packets': [2, 6, 2, 2, 2, 5, 2, 2, 4, 4],
    'Payment': [386, 289, 393, 110, 280, 167, 271, 274, 148, 198],
    'High_Value': ['Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'No']
}

df_customer = pd.DataFrame(customer_data)
print("Customer Data:")
print(df_customer)

# Prepare data for classification
X_customer = df_customer[['Candies', 'Mangoes', 'Milk_Packets', 'Payment']].values
y_customer = (df_customer['High_Value'] == 'Yes').astype(int).values

# Normalize features for better convergence
X_customer_norm = (X_customer - X_customer.mean(axis=0)) / X_customer.std(axis=0)

# Train perceptron with sigmoid activation
customer_perceptron = CustomPerceptron(learning_rate=0.1, activation='sigmoid', max_epochs=1000)
customer_perceptron.fit(X_customer_norm, y_customer)

# Make predictions
customer_predictions = customer_perceptron.predict(X_customer_norm)
customer_predictions_binary = (customer_predictions > 0.5).astype(int)

print("\nCustomer Classification Results:")
for i in range(len(df_customer)):
    print(f"Customer {df_customer.iloc[i]['Customer']}: Target={y_customer[i]}, Prediction={customer_predictions_binary[i]}")

accuracy = np.mean(customer_predictions_binary == y_customer)
print(f"\nAccuracy: {accuracy:.2f}")

customer_perceptron.plot_error("Customer Data Classification")

# A7. Matrix Pseudo-Inverse Comparison
print("\nA7. Matrix Pseudo-Inverse Comparison")
print("=" * 50)

# Add bias column to input
X_customer_bias = np.column_stack([np.ones(len(X_customer_norm)), X_customer_norm])

# Calculate pseudo-inverse solution
pseudo_inverse_weights = np.linalg.pinv(X_customer_bias) @ y_customer
print("Pseudo-inverse weights:", pseudo_inverse_weights)

# Make predictions with pseudo-inverse
pseudo_predictions = X_customer_bias @ pseudo_inverse_weights
pseudo_predictions_binary = (pseudo_predictions > 0.5).astype(int)

print("\nPseudo-inverse Classification Results:")
for i in range(len(df_customer)):
    print(f"Customer {df_customer.iloc[i]['Customer']}: Target={y_customer[i]}, Prediction={pseudo_predictions_binary[i]}")

pseudo_accuracy = np.mean(pseudo_predictions_binary == y_customer)
print(f"\nPseudo-inverse Accuracy: {pseudo_accuracy:.2f}")

print(f"Perceptron Accuracy: {accuracy:.2f}")
print(f"Pseudo-inverse Accuracy: {pseudo_accuracy:.2f}")

# A8. Multi-Layer Perceptron with Backpropagation
print("\nA8. Multi-Layer Perceptron with Backpropagation for AND Gate")
print("=" * 50)

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.05):
        self.learning_rate = learning_rate
        
        # Initialize weights randomly
        self.W1 = np.random.uniform(-0.5, 0.5, (input_size, hidden_size))
        self.b1 = np.random.uniform(-0.5, 0.5, (1, hidden_size))
        self.W2 = np.random.uniform(-0.5, 0.5, (hidden_size, output_size))
        self.b2 = np.random.uniform(-0.5, 0.5, (1, output_size))
        
        self.errors = []
    
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        m = X.shape[0]
        
        # Calculate gradients for output layer
        dz2 = output - y.reshape(-1, 1)
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Calculate gradients for hidden layer
        dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update weights
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    def train(self, X, y, epochs=1000, tolerance=0.002):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            
            # Calculate error
            error = np.mean((y.reshape(-1, 1) - output) ** 2)
            self.errors.append(error)
            
            if error <= tolerance:
                print(f"MLP converged after {epoch + 1} epochs")
                return epoch + 1
        
        print(f"MLP did not converge within {epochs} epochs")
        return epochs
    
    def predict(self, X):
        return self.forward(X)

# Train MLP for AND gate
mlp_and = MLP(input_size=2, hidden_size=3, output_size=1, learning_rate=0.05)
mlp_epochs = mlp_and.train(X_and.astype(float), y_and.astype(float))

# Test MLP predictions
mlp_predictions = mlp_and.predict(X_and.astype(float))
mlp_predictions_binary = (mlp_predictions > 0.5).astype(int).flatten()

print("\nMLP AND Gate Predictions:")
for i in range(len(X_and)):
    print(f"Input: {X_and[i]}, Target: {y_and[i]}, Prediction: {mlp_predictions_binary[i]}")

# Plot MLP training error
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(mlp_and.errors) + 1), mlp_and.errors, 'r-', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('MLP Training Error for AND Gate')
plt.grid(True, alpha=0.3)
plt.show()

# A9. MLP for XOR Gate
print("\nA9. MLP for XOR Gate")
print("=" * 50)

mlp_xor = MLP(input_size=2, hidden_size=3, output_size=1, learning_rate=0.05)
mlp_xor_epochs = mlp_xor.train(X_xor.astype(float), y_xor.astype(float))

# Test MLP predictions for XOR
mlp_xor_predictions = mlp_xor.predict(X_xor.astype(float))
mlp_xor_predictions_binary = (mlp_xor_predictions > 0.5).astype(int).flatten()

print("\nMLP XOR Gate Predictions:")
for i in range(len(X_xor)):
    print(f"Input: {X_xor[i]}, Target: {y_xor[i]}, Prediction: {mlp_xor_predictions_binary[i]}")

# A11. Scikit-learn MLPClassifier
print("\nA11. Scikit-learn MLPClassifier")
print("=" * 50)

# AND Gate with MLPClassifier
mlp_sklearn_and = MLPClassifier(hidden_layer_sizes=(3,), activation='logistic', 
                               learning_rate_init=0.05, max_iter=1000, random_state=42)
mlp_sklearn_and.fit(X_and, y_and)

sklearn_and_predictions = mlp_sklearn_and.predict(X_and)
print("Scikit-learn MLP AND Gate Predictions:")
for i in range(len(X_and)):
    print(f"Input: {X_and[i]}, Target: {y_and[i]}, Prediction: {sklearn_and_predictions[i]}")

# XOR Gate with MLPClassifier
mlp_sklearn_xor = MLPClassifier(hidden_layer_sizes=(3,), activation='logistic', 
                               learning_rate_init=0.05, max_iter=1000, random_state=42)
mlp_sklearn_xor.fit(X_xor, y_xor)

sklearn_xor_predictions = mlp_sklearn_xor.predict(X_xor)
print("\nScikit-learn MLP XOR Gate Predictions:")
for i in range(len(X_xor)):
    print(f"Input: {X_xor[i]}, Target: {y_xor[i]}, Prediction: {sklearn_xor_predictions[i]}")

# A12. MLPClassifier on Elephant Calls Dataset
print("\nA12. MLPClassifier on Elephant Calls Dataset")
print("=" * 50)

# Load elephant calls dataset (assuming it's in the same directory)
try:
    # Try to read the CSV file
    elephant_data = pd.read_csv('elephant_calls.csv')
    print("Elephant calls dataset loaded successfully!")
    print(f"Dataset shape: {elephant_data.shape}")
    print(f"Columns: {elephant_data.columns.tolist()}")
    
    # For this example, let's use some acoustic features to predict call type
    # You can modify this based on your specific requirements
    
    # Select relevant features (modify as needed)
    feature_columns = ['Distance', 'CallerAge', 'Elicitor1Age', 'sprsMed', 'sprsMbw', 'sprsEqbw', 'F1', 'F2', 'F3', 'F4']
    available_features = [col for col in feature_columns if col in elephant_data.columns]
    
    if len(available_features) > 0:
        X_elephant = elephant_data[available_features].dropna()
        
        # Create a binary classification target (you can modify this)
        # For example, classify based on Distance (close vs far calls)
        if 'Distance' in elephant_data.columns:
            y_elephant = (elephant_data['Distance'].fillna(elephant_data['Distance'].median()) > 
                         elephant_data['Distance'].fillna(elephant_data['Distance'].median()).median()).astype(int)
            y_elephant = y_elephant[X_elephant.index]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_elephant, y_elephant, 
                                                              test_size=0.3, random_state=42)
            
            # Train MLPClassifier
            elephant_mlp = MLPClassifier(hidden_layer_sizes=(10, 5), activation='relu', 
                                       learning_rate_init=0.01, max_iter=1000, random_state=42)
            elephant_mlp.fit(X_train, y_train)
            
            # Make predictions
            y_pred = elephant_mlp.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"\nElephant Calls Classification Results:")
            print(f"Features used: {available_features}")
            print(f"Training samples: {len(X_train)}")
            print(f"Test samples: {len(X_test)}")
            print(f"Accuracy: {accuracy:.3f}")
            print(f"\nClassification Report:")
            print(classification_report(y_test, y_pred))
        else:
            print("Distance column not found for creating target variable")
    else:
        print("Required feature columns not found in the dataset")
        
except FileNotFoundError:
    print("Elephant calls CSV file not found in the current directory.")
    print("Please ensure the CSV file is in the same directory as this script.")
except Exception as e:
    print(f"Error loading elephant calls dataset: {e}")

print("\n" + "="*60)
print("SUMMARY OF RESULTS")
print("="*60)

print(f"\nA2. AND Gate Step Activation: {perceptron_and_step.epochs_to_converge} epochs")
print(f"\nA3. AND Gate Activation Function Comparison:")
for activation, epochs in convergence_results.items():
    print(f"  {activation}: {epochs} epochs")

print(f"\nA4. Best learning rate for AND gate: {min(lr_convergence, key=lr_convergence.get)} (converged in {min(lr_convergence.values())} epochs)")

print(f"\nA5. XOR Gate Results:")
for activation, epochs in xor_convergence.items():
    print(f"  {activation}: {epochs} epochs")

print(f"\nA6. Customer Classification Accuracy: {accuracy:.3f}")
print(f"A7. Pseudo-inverse Accuracy: {pseudo_accuracy:.3f}")
print(f"\nA8. MLP AND Gate: {mlp_epochs} epochs")
print(f"A9. MLP XOR Gate: {mlp_xor_epochs} epochs")
