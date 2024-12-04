import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train, x_test = x_train/255.0, x_test/255.0

# Reshape to include the channel (CNNs)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print(x_train.shape, x_test.shape)

# Create the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), 
    layers.MaxPooling2D((2, 2)), 
    layers.Conv2D(64, (3, 3), activation='relu'), 
    layers.MaxPooling2D((2, 2)), 
    layers.Flatten(),
    layers.Dense(64, activation='relu'),  
    layers.Dense(10, activation='softmax') 
])

# Compile the model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy']) 

# Train the model
history = model.fit(x_train, y_train, epochs=5, validation_split=0.2) 

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)  
print(f"Accuracy: {test_acc}")

predictions = model.predict(x_test) 

# Visualize an example
plt.imshow(x_test[0].reshape(28, 28), cmap='gray') 
plt.title(f"Prediction: {predictions[0].argmax()}") 
plt.show()