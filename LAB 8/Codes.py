import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# A1. Basic functions for perceptron components
class PerceptronComponents:
    @staticmethod
    def summation_unit(inputs, weights):
        """Summation unit: calculates weighted sum + bias"""
        return np.dot(inputs, weights[1:]) + weights[0]
    
    @staticmethod
    def step_activation(x):
        """Step activation function"""
        return 1 if x >= 0 else 0
    
    @staticmethod
    def bipolar_step_activation(x):
        """Bipolar step activation function"""
        return 1 if x >= 0 else -1
    
    @staticmethod
    def sigmoid_activation(x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def tanh_activation(x):
        """Hyperbolic tangent activation function"""
        return np.tanh(x)
    
    @staticmethod
    def relu_activation(x):
        """ReLU activation function"""
        return max(0, x)
    
    @staticmethod
    def leaky_relu_activation(x, alpha=0.01):
        """Leaky ReLU activation function"""
        return max(alpha * x, x)
    
    @staticmethod
    def comparator_unit(target, output):
        """Error calculation unit"""
        return target - output

# Custom Perceptron Implementation
class CustomPerceptron:
    def __init__(self, learning_rate=0.05, max_epochs=1000, convergence_error=0.002):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.convergence_error = convergence_error
        self.weights = None
        self.errors = []
        self.epochs_to_converge = 0
        
    def initialize_weights(self, n_features, initial_weights=None):
        """Initialize weights"""
        if initial_weights is not None:
            self.weights = np.array(initial_weights)
        else:
            self.weights = np.random.uniform(-0.5, 0.5, n_features + 1)
    
    def predict_single(self, x, activation_func):
        """Predict single sample"""
        net_input = PerceptronComponents.summation_unit(x, self.weights)
        return activation_func(net_input)
    
    def fit(self, X, y, activation_func, initial_weights=None):
        """Train the perceptron"""
        self.initialize_weights(X.shape[1], initial_weights)
        self.errors = []
        
        for epoch in range(self.max_epochs):
            epoch_error = 0
            for xi, target in zip(X, y):
                prediction = self.predict_single(xi, activation_func)
                error = PerceptronComponents.comparator_unit(target, prediction)
                
                # Update weights
                self.weights[0] += self.learning_rate * error  # bias
                self.weights[1:] += self.learning_rate * error * xi
                
                epoch_error += error ** 2
            
            # Calculate sum square error
            sse = epoch_error / len(X)
            self.errors.append(sse)
            
            if sse <= self.convergence_error:
                self.epochs_to_converge = epoch + 1
                print(f"Converged after {self.epochs_to_converge} epochs")
                break
        else:
            self.epochs_to_converge = self.max_epochs
            print(f"Did not converge within {self.max_epochs} epochs")
    
    def predict(self, X, activation_func):
        """Predict multiple samples"""
        return [self.predict_single(xi, activation_func) for xi in X]

# A2. AND Gate Logic Implementation
def run_and_gate_experiment(activation_func, activation_name, learning_rate=0.05):
    """Run AND gate experiment with specified activation function"""
    print(f"\n=== AND Gate with {activation_name} Activation ===")
    
    # AND gate training data
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([0, 0, 0, 1])
    
    # Initialize perceptron
    perceptron = CustomPerceptron(learning_rate=learning_rate)
    initial_weights = [10, 0.2, -0.75]  # w0, w1, w2
    
    # Train
    perceptron.fit(X_and, y_and, activation_func, initial_weights)
    
    # Test
    predictions = perceptron.predict(X_and, activation_func)
    print(f"Predictions: {predictions}")
    print(f"Targets:     {y_and}")
    print(f"Final weights: {perceptron.weights}")
    
    return perceptron.epochs_to_converge, perceptron.errors

# A3. Compare different activation functions
def compare_activation_functions():
    """Compare convergence for different activation functions"""
    print("\n" + "="*50)
    print("A3. COMPARING ACTIVATION FUNCTIONS")
    print("="*50)
    
    activation_functions = {
        'Step': PerceptronComponents.step_activation,
        'Bipolar Step': PerceptronComponents.bipolar_step_activation,
        'Sigmoid': PerceptronComponents.sigmoid_activation,
        'ReLU': PerceptronComponents.relu_activation
    }
    
    results = {}
    plt.figure(figsize=(12, 8))
    
    for i, (name, func) in enumerate(activation_functions.items()):
        epochs, errors = run_and_gate_experiment(func, name)
        results[name] = epochs
        
        plt.subplot(2, 2, i+1)
        plt.plot(errors)
        plt.title(f'{name} Activation - Epochs: {epochs}')
        plt.xlabel('Epoch')
        plt.ylabel('Sum Square Error')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('activation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

# A4. Learning rate comparison
def compare_learning_rates():
    """Compare convergence for different learning rates"""
    print("\n" + "="*50)
    print("A4. COMPARING LEARNING RATES")
    print("="*50)
    
    learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    epochs_list = []
    
    for lr in learning_rates:
        print(f"\nTesting learning rate: {lr}")
        epochs, _ = run_and_gate_experiment(PerceptronComponents.step_activation, "Step", lr)
        epochs_list.append(epochs)
    
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, epochs_list, 'bo-')
    plt.xlabel('Learning Rate')
    plt.ylabel('Epochs to Converge')
    plt.title('Learning Rate vs Convergence Speed (AND Gate)')
    plt.grid(True)
    plt.savefig('learning_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return dict(zip(learning_rates, epochs_list))

# A5. XOR Gate Logic Implementation
def run_xor_gate_experiments():
    """Run XOR gate experiments with different activation functions"""
    print("\n" + "="*50)
    print("A5. XOR GATE EXPERIMENTS")
    print("="*50)
    
    # XOR gate training data
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
    
    activation_functions = {
        'Step': PerceptronComponents.step_activation,
        'Bipolar Step': PerceptronComponents.bipolar_step_activation,
        'Sigmoid': PerceptronComponents.sigmoid_activation,
        'ReLU': PerceptronComponents.relu_activation
    }
    
    results = {}
    for name, func in activation_functions.items():
        print(f"\n--- XOR Gate with {name} Activation ---")
        perceptron = CustomPerceptron(learning_rate=0.05)
        initial_weights = [10, 0.2, -0.75]
        
        perceptron.fit(X_xor, y_xor, func, initial_weights)
        predictions = perceptron.predict(X_xor, func)
        
        print(f"Predictions: {predictions}")
        print(f"Targets:     {y_xor}")
        print(f"Epochs to converge: {perceptron.epochs_to_converge}")
        
        results[name] = {
            'epochs': perceptron.epochs_to_converge,
            'predictions': predictions,
            'accuracy': sum(p == t for p, t in zip(predictions, y_xor)) / len(y_xor)
        }
    
    return results

# A6. Customer data classification
def customer_classification():
    """Classify customer transactions as high/low value"""
    print("\n" + "="*50)
    print("A6. CUSTOMER DATA CLASSIFICATION")
    print("="*50)
    
    # Customer data from the problem
    customer_data = {
        'Candies': [20, 16, 27, 19, 24, 22, 15, 18, 21, 16],
        'Mangoes': [6, 3, 6, 1, 4, 1, 4, 4, 1, 2],
        'Milk_Packets': [2, 6, 2, 2, 2, 5, 2, 2, 4, 4],
        'Payment': [386, 289, 393, 110, 280, 167, 271, 274, 148, 198],
        'High_Value': [1, 1, 1, 0, 1, 0, 1, 1, 0, 0]  # Yes=1, No=0
    }
    
    df = pd.DataFrame(customer_data)
    print("Customer Data:")
    print(df)
    
    # Prepare data
    X = df[['Candies', 'Mangoes', 'Milk_Packets', 'Payment']].values
    y = df['High_Value'].values
    
    # Normalize features
    X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)
    
    # Train perceptron with sigmoid activation
    perceptron = CustomPerceptron(learning_rate=0.1, max_epochs=1000)
    perceptron.fit(X_normalized, y, PerceptronComponents.sigmoid_activation)
    
    # Predictions
    predictions = perceptron.predict(X_normalized, PerceptronComponents.sigmoid_activation)
    accuracy = sum(p == t for p, t in zip(predictions, y)) / len(y)
    
    print(f"\nPerceptron Results:")
    print(f"Predictions: {predictions}")
    print(f"Actual:      {list(y)}")
    print(f"Accuracy:    {accuracy:.2%}")
    
    return X_normalized, y, perceptron

# A7. Matrix pseudo-inverse comparison
def pseudo_inverse_comparison(X, y):
    """Compare perceptron results with pseudo-inverse method"""
    print("\n" + "="*50)
    print("A7. PSEUDO-INVERSE COMPARISON")
    print("="*50)
    
    # Add bias column
    X_bias = np.column_stack([np.ones(X.shape[0]), X])
    
    # Pseudo-inverse solution
    weights_pinv = np.linalg.pinv(X_bias) @ y
    
    # Predictions using pseudo-inverse weights
    predictions_pinv = []
    for xi in X:
        net_input = PerceptronComponents.summation_unit(xi, weights_pinv)
        pred = PerceptronComponents.sigmoid_activation(net_input)
        predictions_pinv.append(1 if pred >= 0.5 else 0)
    
    accuracy_pinv = sum(p == t for p, t in zip(predictions_pinv, y)) / len(y)
    
    print(f"Pseudo-inverse weights: {weights_pinv}")
    print(f"Pseudo-inverse predictions: {predictions_pinv}")
    print(f"Pseudo-inverse accuracy: {accuracy_pinv:.2%}")
    
    return weights_pinv, predictions_pinv

# Neural Network Implementation for Backpropagation
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.05):
        self.learning_rate = learning_rate
        
        # Initialize weights randomly
        self.W1 = np.random.uniform(-0.5, 0.5, (input_size + 1, hidden_size))  # +1 for bias
        self.W2 = np.random.uniform(-0.5, 0.5, (hidden_size + 1, output_size))  # +1 for bias
        
        self.errors = []
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        # Add bias to input
        X_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # Hidden layer
        self.hidden_input = X_bias @ self.W1
        self.hidden_output = self.sigmoid(self.hidden_input)
        
        # Add bias to hidden output
        hidden_bias = np.column_stack([np.ones(self.hidden_output.shape[0]), self.hidden_output])
        
        # Output layer
        self.output_input = hidden_bias @ self.W2
        self.output = self.sigmoid(self.output_input)
        
        return self.output
    
    def backward(self, X, y):
        m = X.shape[0]
        
        # Calculate output layer error
        output_error = y.reshape(-1, 1) - self.output
        output_delta = output_error * self.sigmoid_derivative(self.output)
        
        # Calculate hidden layer error
        hidden_bias = np.column_stack([np.ones(self.hidden_output.shape[0]), self.hidden_output])
        hidden_error = output_delta @ self.W2[1:].T  # Exclude bias weights
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)
        
        # Update weights
        X_bias = np.column_stack([np.ones(X.shape[0]), X])
        self.W2 += self.learning_rate * hidden_bias.T @ output_delta
        self.W1 += self.learning_rate * X_bias.T @ hidden_delta
    
    def train(self, X, y, epochs=1000, convergence_error=0.002):
        self.errors = []
        
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Calculate error
            error = np.mean((y.reshape(-1, 1) - output) ** 2)
            self.errors.append(error)
            
            # Backward pass
            self.backward(X, y)
            
            if error <= convergence_error:
                print(f"Neural Network converged after {epoch + 1} epochs")
                return epoch + 1
        
        print(f"Neural Network did not converge within {epochs} epochs")
        return epochs

# A8 & A9. Neural Network experiments
def neural_network_experiments():
    """Run neural network experiments for AND and XOR gates"""
    print("\n" + "="*50)
    print("A8 & A9. NEURAL NETWORK EXPERIMENTS")
    print("="*50)
    
    # AND Gate
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([0, 0, 0, 1])
    
    print("A8. Neural Network - AND Gate")
    nn_and = NeuralNetwork(2, 2, 1, learning_rate=0.05)
    epochs_and = nn_and.train(X_and, y_and)
    
    predictions_and = nn_and.forward(X_and)
    pred_binary_and = (predictions_and.flatten() >= 0.5).astype(int)
    print(f"AND Gate Predictions: {pred_binary_and}")
    print(f"AND Gate Targets:     {y_and}")
    
    # XOR Gate
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
    
    print("\nA9. Neural Network - XOR Gate")
    nn_xor = NeuralNetwork(2, 2, 1, learning_rate=0.05)
    epochs_xor = nn_xor.train(X_xor, y_xor)
    
    predictions_xor = nn_xor.forward(X_xor)
    pred_binary_xor = (predictions_xor.flatten() >= 0.5).astype(int)
    print(f"XOR Gate Predictions: {pred_binary_xor}")
    print(f"XOR Gate Targets:     {y_xor}")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(nn_and.errors)
    plt.title(f'AND Gate Training (Converged: {epochs_and} epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(nn_xor.errors)
    plt.title(f'XOR Gate Training (Converged: {epochs_xor} epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('neural_network_training.png', dpi=300, bbox_inches='tight')
    plt.show()

# A10. Two output nodes implementation
class TwoOutputPerceptron:
    def __init__(self, learning_rate=0.05, max_epochs=1000, convergence_error=0.002):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.convergence_error = convergence_error
        self.weights1 = None  # Weights for output node 1
        self.weights2 = None  # Weights for output node 2
        self.errors = []
        self.epochs_to_converge = 0
    
    def initialize_weights(self, n_features):
        """Initialize weights for both output nodes"""
        self.weights1 = np.random.uniform(-0.5, 0.5, n_features + 1)
        self.weights2 = np.random.uniform(-0.5, 0.5, n_features + 1)
    
    def predict_single(self, x, activation_func):
        """Predict single sample with two outputs"""
        net_input1 = PerceptronComponents.summation_unit(x, self.weights1)
        net_input2 = PerceptronComponents.summation_unit(x, self.weights2)
        
        output1 = activation_func(net_input1)
        output2 = activation_func(net_input2)
        
        return [output1, output2]
    
    def fit(self, X, y_binary, activation_func):
        """Train the two-output perceptron"""
        self.initialize_weights(X.shape[1])
        self.errors = []
        
        # Convert binary targets to two-output format
        # 0 -> [1, 0], 1 -> [0, 1]
        y_two_output = []
        for target in y_binary:
            if target == 0:
                y_two_output.append([1, 0])
            else:
                y_two_output.append([0, 1])
        y_two_output = np.array(y_two_output)
        
        for epoch in range(self.max_epochs):
            epoch_error = 0
            
            for xi, target in zip(X, y_two_output):
                predictions = self.predict_single(xi, activation_func)
                
                # Calculate errors for both outputs
                error1 = target[0] - predictions[0]
                error2 = target[1] - predictions[1]
                
                # Update weights for output node 1
                self.weights1[0] += self.learning_rate * error1  # bias
                self.weights1[1:] += self.learning_rate * error1 * xi
                
                # Update weights for output node 2
                self.weights2[0] += self.learning_rate * error2  # bias
                self.weights2[1:] += self.learning_rate * error2 * xi
                
                epoch_error += error1**2 + error2**2
            
            # Calculate sum square error
            sse = epoch_error / (2 * len(X))  # Divide by 2*n because we have 2 outputs
            self.errors.append(sse)
            
            if sse <= self.convergence_error:
                self.epochs_to_converge = epoch + 1
                print(f"Two-output perceptron converged after {self.epochs_to_converge} epochs")
                break
        else:
            self.epochs_to_converge = self.max_epochs
            print(f"Two-output perceptron did not converge within {self.max_epochs} epochs")
    
    def predict(self, X, activation_func):
        """Predict multiple samples with two outputs"""
        predictions = []
        for xi in X:
            pred = self.predict_single(xi, activation_func)
            predictions.append(pred)
        return np.array(predictions)
    
    def convert_to_binary(self, two_output_predictions):
        """Convert two-output format back to binary"""
        binary_predictions = []
        for pred in two_output_predictions:
            # Choose the output with higher value
            if pred[0] > pred[1]:
                binary_predictions.append(0)  # [1,0] -> 0
            else:
                binary_predictions.append(1)  # [0,1] -> 1
        return np.array(binary_predictions)

def run_two_output_experiments():
    """Run experiments with two output nodes (A10)"""
    print("\n" + "="*50)
    print("A10. TWO OUTPUT NODES EXPERIMENTS")
    print("="*50)
    
    # AND Gate experiment
    print("Two-Output AND Gate:")
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([0, 0, 0, 1])
    
    perceptron_and = TwoOutputPerceptron(learning_rate=0.05)
    perceptron_and.fit(X_and, y_and, PerceptronComponents.step_activation)
    
    # Get predictions
    predictions_two_output = perceptron_and.predict(X_and, PerceptronComponents.step_activation)
    predictions_binary = perceptron_and.convert_to_binary(predictions_two_output)
    
    print(f"Input -> Two-Output -> Binary")
    for i, (inp, two_out, binary) in enumerate(zip(X_and, predictions_two_output, predictions_binary)):
        print(f"{inp} -> [{two_out[0]:.0f}, {two_out[1]:.0f}] -> {binary} (target: {y_and[i]})")
    
    accuracy_and = sum(predictions_binary == y_and) / len(y_and)
    print(f"AND Gate Accuracy: {accuracy_and:.2%}")
    print(f"Epochs to converge: {perceptron_and.epochs_to_converge}")
    
    # XOR Gate experiment
    print("\nTwo-Output XOR Gate:")
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
    
    perceptron_xor = TwoOutputPerceptron(learning_rate=0.05)
    perceptron_xor.fit(X_xor, y_xor, PerceptronComponents.step_activation)
    
    # Get predictions
    predictions_two_output_xor = perceptron_xor.predict(X_xor, PerceptronComponents.step_activation)
    predictions_binary_xor = perceptron_xor.convert_to_binary(predictions_two_output_xor)
    
    print(f"Input -> Two-Output -> Binary")
    for i, (inp, two_out, binary) in enumerate(zip(X_xor, predictions_two_output_xor, predictions_binary_xor)):
        print(f"{inp} -> [{two_out[0]:.0f}, {two_out[1]:.0f}] -> {binary} (target: {y_xor[i]})")
    
    accuracy_xor = sum(predictions_binary_xor == y_xor) / len(y_xor)
    print(f"XOR Gate Accuracy: {accuracy_xor:.2%}")
    print(f"Epochs to converge: {perceptron_xor.epochs_to_converge}")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(perceptron_and.errors)
    plt.title(f'Two-Output AND Gate Training\n(Converged: {perceptron_and.epochs_to_converge} epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Sum Square Error')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(perceptron_xor.errors)
    plt.title(f'Two-Output XOR Gate Training\n(Converged: {perceptron_xor.epochs_to_converge} epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Sum Square Error')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('two_output_training.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'AND': {'accuracy': accuracy_and, 'epochs': perceptron_and.epochs_to_converge},
        'XOR': {'accuracy': accuracy_xor, 'epochs': perceptron_xor.epochs_to_converge}
    }

# A11. Scikit-learn MLPClassifier
def sklearn_mlp_experiments():
    """Use scikit-learn MLPClassifier for AND and XOR gates"""
    print("\n" + "="*50)
    print("A11. SCIKIT-LEARN MLP EXPERIMENTS")
    print("="*50)
    
    # Data
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([0, 0, 0, 1])
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
    
    # AND Gate
    print("AND Gate with MLPClassifier:")
    mlp_and = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000, 
                           learning_rate_init=0.05, random_state=42)
    mlp_and.fit(X_and, y_and)
    pred_and = mlp_and.predict(X_and)
    print(f"Predictions: {pred_and}")
    print(f"Accuracy: {accuracy_score(y_and, pred_and):.2%}")
    
    # XOR Gate
    print("\nXOR Gate with MLPClassifier:")
    mlp_xor = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000,
                           learning_rate_init=0.05, random_state=42)
    mlp_xor.fit(X_xor, y_xor)
    pred_xor = mlp_xor.predict(X_xor)
    print(f"Predictions: {pred_xor}")
    print(f"Accuracy: {accuracy_score(y_xor, pred_xor):.2%}")

# A12. MLPClassifier on elephant dataset
def elephant_dataset_analysis():
    """Apply MLPClassifier to elephant calls dataset"""
    print("\n" + "="*50)
    print("A12. ELEPHANT DATASET ANALYSIS")
    print("="*50)
    

    file_path = r"D:\\OneDrive - Amrita vishwa vidyapeetham\\SEM 5\\23CSE301 Machine Learning\\LAB\\Machine-Learning\\LAB 3\\20231225_dfall_obs_data_and_spectral_features_revision1_n469.csv"
    df = pd.read_csv(file_path)

    print(f"Dataset shape: {df.shape}")
    print("\nDataset info:")
    print(df.info())
       
    # Select relevant features for classification
    # Using some acoustic features and trying to predict Context2
    feature_columns = ['sprsMed', 'sprsMbw', 'sprsEqbw', 'sprsMc', 'CallerAge', 'Elicitor1Age']
    target_column = 'Context2'  # or another suitable target
   
    # Check if columns exist
    available_features = [col for col in feature_columns if col in df.columns]
        
    if len(available_features) < 2:
        print("Not enough suitable features found in dataset")
        return
        
    # Prepare data
    X = df[available_features].fillna(0)  # Fill missing values
    y = df[target_column] if target_column in df.columns else df['Call_Type']
        
    # Encode categorical target if needed
    if y.dtype == 'object':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)
        
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
       
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
        
    # Train MLP
    mlp = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
    mlp.fit(X_train_scaled, y_train)
        
    # Predictions
    y_pred = mlp.predict(X_test_scaled)
        
    print(f"\nElephant Dataset MLP Results:")
    print(f"Features used: {available_features}")
    print(f"Target: {target_column}")
    print(f"Training accuracy: {mlp.score(X_train_scaled, y_train):.3f}")
    print(f"Testing accuracy: {mlp.score(X_test_scaled, y_test):.3f}")
    print(f"Number of iterations: {mlp.n_iter_}")
        
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
        
# Main execution function
def main():
    """Run all experiments"""
    print("PERCEPTRON AND NEURAL NETWORK EXPERIMENTS")
    print("=" * 60)
    
    # A2. Basic AND gate with step activation
    print("\nA2. BASIC AND GATE EXPERIMENT")
    run_and_gate_experiment(PerceptronComponents.step_activation, "Step")
    
    # A3. Compare activation functions
    activation_results = compare_activation_functions()
    print(f"\nActivation function comparison results: {activation_results}")
    
    # A4. Compare learning rates
    lr_results = compare_learning_rates()
    print(f"\nLearning rate comparison results: {lr_results}")
    
    # A5. XOR gate experiments
    xor_results = run_xor_gate_experiments()
    print(f"\nXOR gate results summary:")
    for name, result in xor_results.items():
        print(f"{name}: {result['epochs']} epochs, {result['accuracy']:.2%} accuracy")
    
    # A6. Customer classification
    X_norm, y_customer, customer_perceptron = customer_classification()
    
    # A7. Pseudo-inverse comparison
    pseudo_inverse_comparison(X_norm, y_customer)
    
    # A8 & A9. Neural network experiments
    neural_network_experiments()
    
    # A10. Two output nodes experiments
    two_output_results = run_two_output_experiments()
    print(f"\nTwo-output perceptron results: {two_output_results}")
    
    # A11. Scikit-learn MLP
    sklearn_mlp_experiments()
    
    # A12. Elephant dataset
    elephant_dataset_analysis()
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    main()