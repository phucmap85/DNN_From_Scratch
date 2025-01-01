import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.utils import to_categorical

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

X = np.concatenate((train_images, test_images))
Y = np.concatenate((train_labels, test_labels))

# Normalize the images to a range of 0 to 1
X = X / 255.0

# Perform balanced train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

# Function to select a balanced subset of size 3000
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

# Shuffle array with random_state = 42
np.random.seed(42)
indices = np.arange(len(Y_subset))
np.random.shuffle(indices)

X_subset = X_subset[indices]
Y_subset = Y_subset[indices]

# Build the model
model = Sequential([
    Flatten(input_shape=(28, 28)),        # Flatten the 28x28 images into 1D vectors
    Dense(16, activation='relu'),         # Hidden layer with 16 nodes and ReLU activation
    Dense(16, activation='relu'),
    Dense(10, activation='softmax')       # Output layer with 10 nodes (digits 0-9) and softmax activation
])

# Compile the model
model.compile(optimizer='sgd', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

# Train the model
history = model.fit(X_subset, Y_subset, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'Test Accuracy: {test_accuracy:.4f}')