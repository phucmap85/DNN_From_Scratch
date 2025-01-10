import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from collections import Counter

# Load MNIST dataset
(X, Y), (_, _) = mnist.load_data()

# Flatten the images
X = X.reshape(X.shape[0], -1).astype('float32') / 255.0

# Perform balanced train-test split
test_size = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, stratify=Y, random_state=42)

# Function to select a balanced subset
def get_balanced_subset(X, Y, size):
    unique_classes = np.unique(Y)
    samples_per_class = size // len(unique_classes)
    selected_indices = []
    
    for label in unique_classes:
        indices = np.where(Y == label)[0]
        selected_indices.extend(indices[:samples_per_class])
    
    return X[selected_indices], Y[selected_indices]

X_subset, Y_subset = get_balanced_subset(X_train, Y_train, size=5000)

# Print the distribution of labels in the subset
label_counts = Counter(Y_subset)
print("Label distribution in the subset:")
for label, count in label_counts.items():
    print(f"Label {label}: {count} samples")

# Combine X_subset and Y_subset for saving
TT_subset = np.concatenate((X_subset, Y_subset.reshape(len(Y_subset), 1)), axis=1)

# Shuffle the array before saving with random_state = 42
np.random.seed(42)
np.random.shuffle(TT_subset)
np.savetxt("MAIN.INP", TT_subset, fmt='%.14f')

# Write the size of dataset, the dimension of the first element of X_subset and the train_size to the first line of MAIN.INP
with open("MAIN.INP", "r+") as file:
    content = file.read()
    file.seek(0, 0)
    file.write(f"{TT_subset.shape[0]} {X_subset.shape[1]} {1 - test_size}\n" + content)