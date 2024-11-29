import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate XOR-like dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[1], [0], [0], [1]], dtype=float)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(2,)),  # Input layer for 2 features
    tf.keras.layers.Dense(4, activation='relu'),   # Hidden layer with 4 units
    tf.keras.layers.Dense(2, activation='relu'),   # Hidden layer with 4 units
    tf.keras.layers.Dense(1, activation='sigmoid') # Output layer with 1 unit (binary output)
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), 
    loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X, y, batch_size=2, epochs=100)

# Evaluate the model
loss, accuracy = model.evaluate(X, y)
print(f"Final accuracy: {accuracy * 100:.2f}%")

# Plot training loss over epochs
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Test the model with the XOR input
predictions = model.predict(X)
print("\nPredictions:")
print(predictions)
