import random
from statistics import mean, median, mode
numbers = [random.randint(100, 150) for _ in range(100)]
print("Numbers:", numbers)
print("Mean:", mean(numbers))
print("Median:", median(numbers))
print("Mode:", mode(numbers))