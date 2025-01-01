import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter

df = pd.read_csv("mnist_train.csv")

Y = df['label'].to_numpy()

T = df.drop('label', axis=1)

X = T.to_numpy()
X = X / 255.0

# Perform balanced train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

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