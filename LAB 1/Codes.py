# Q1
def count_vowels_consonants(s):
    vowels = "aeiouAEIOU"                                       # List of vowels
    vowel_count = 0                                             # Initialize vowel count
    consonant_count = 0                                         # Initialize consonant count
    for char in s:                                              # Iterate through each character in the string                              
        if char.isalpha():                                      # Check if the character is a letter
            if char in vowels:                                  # Check if the character is a vowel
                vowel_count += 1                                # Increment vowel count
            else:
                consonant_count += 1                            # Increment consonant count
    return vowel_count, consonant_count                         # Return the counts of vowels and consonants
input_str = input("Enter a string: ")                           # Input string from user
vowels, consonants = count_vowels_consonants(input_str)         # Call the function to count vowels and consonants
print(f"Number of vowels: {vowels}")
print(f"Number of consonants: {consonants}")

print("----------------------------------------------------------------------------------------------------------")

#Q2
def matrix_multiply(A, B):                                                              
    if len(A[0]) != len(B):                                  # Check if the number of columns in A equals the number of rows in B   
        return "Error: Matrices cannot be multiplied. Columns of A must equal rows of B."
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]                 # Initialize result matrix with zeros
    for i in range(len(A)):                                  # Iterate through each row in A
        for j in range(len(B[0])):                           # Iterate through each element of the result matrix
            for k in range(len(B)):                          # Iterate through each column in B
                result[i][j] += A[i][k] * B[k][j]            # Multiply and accumulate the result
    return result                                            # Return the resulting matrix
def input_matrix(name):
    rows = int(input(f"Enter number of rows for matrix {name}: "))                      # Input number of rows for the matrix
    cols = int(input(f"Enter number of columns for matrix {name}: "))                   # Input number of columns for the matrix
    print(f"Enter elements of matrix {name} row-wise (space separated):")               # Input elements of the matrix
    matrix = []                                                             # Initialize an empty matrix
    for i in range(rows):                                                   # Iterate through each row
        row = list(map(int, input().split()))                               # Input the row elements as integers              
        if len(row) != cols:                                                # Check if the number of elements in the row matches the specified number of columns
            print("Error: Incorrect number of columns entered.")    
            return None                                                     # If not, return None
        matrix.append(row)                                                  # Append the row to the matrix
    return matrix                                   # Return the input matrix
A = input_matrix("A")
B = input_matrix("B")           
if A is not None and B is not None:                 # If both matrices are valid, perform multiplication
    product = matrix_multiply(A, B)                 # Call the matrix multiplication function
    if isinstance(product, str):                    # If an error message is returned
        print(product)                              # Print the error message
    else:
        print("Product AB:")                        # Print the resulting product matrix
        for row in product:
            print(" ".join(map(str, row)))          # Display each row of the product matrix

print("----------------------------------------------------------------------------------------------------------")

# Q3
def count_common_elements(list1, list2):
    set1 = set(list1)                        # Convert list1 to a set for faster lookup 
    set2 = set(list2)                        # Convert list2 to a set for faster lookup
    common = set1 & set2                     # Find common elements using set intersection
    return len(common)                       # Return the number of common elements
list1 = list(map(int, input("Enter integers for list 1 (space separated): ").split()))          # Input integers for list 1
list2 = list(map(int, input("Enter integers for list 2 (space separated): ").split()))          # Input integers for list 2
num_common = count_common_elements(list1, list2)                # Call the function to count common elements
print(f"Number of common elements: {num_common}")               # Display the number of common elements

print("----------------------------------------------------------------------------------------------------------")

# Q4
def transpose_matrix(matrix):
    rows = len(matrix)                                  # Get the number of rows in the matrix 
    cols = len(matrix[0])                               # Get the number of columns in the matrix
    transpose = [[matrix[j][i] for j in range(rows)] for i in range(cols)]          # Create the transposed matrix using list comprehension
    return transpose                                    # Return the transposed matrix
def input_matrix_for_transpose():
    rows = int(input("Enter number of rows for the matrix: "))
    cols = int(input("Enter number of columns for the matrix: "))
    print("Enter elements of the matrix row-wise (space separated):")
    matrix = []                                         # Initialize an empty matrix  
    for _ in range(rows):
        row = list(map(int, input().split()))           # Input the row elements as integers
        if len(row) != cols:                            # Check if the number of elements in the row matches the specified number of columns
            print("Error: Incorrect number of columns entered.")
            return None                                 # If not, return None
        matrix.append(row)                              # Append the row to the matrix  
    return matrix
matrix = input_matrix_for_transpose()                   # Input matrix for transposition
if matrix is not None:
    transposed = transpose_matrix(matrix)               # Call the function to transpose the matrix
    print("Transpose of the matrix:")
    for row in transposed:
        print(" ".join(map(str, row)))                  # Display each row of the transposed matrix

print("----------------------------------------------------------------------------------------------------------")

# Q5
import random
from statistics import mean, median, mode
numbers = [random.randint(100, 150) for _ in range(100)]        # Generating 100 random integers between 100 and 150
print("Numbers:", numbers)                                      # Display the generated numbers
print("Mean:", mean(numbers))                                   # Calculate and display the mean    
print("Median:", median(numbers))                               # Calculate and display the median
print("Mode:", mode(numbers))                                   # Calculate and display the mode

print("----------------------------------------------------------------------------------------------------------")