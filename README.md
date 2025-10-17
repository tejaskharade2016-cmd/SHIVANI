import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# Example values – you should replace these with your actual dataset
IMG_HEIGHT = 32
IMG_WIDTH = 32
NUM_CLASSES = 43  # Number of traffic sign categories in GTSRB

# Load your dataset here (you’ll need to write preprocessing code)
# X_train, y_train = ...
# X_test, y_test = ...

# Normalize images
X_train = X_train / 255.0
X_test = X_test / 255.0

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the model
model.save("traffic_sign_model.h5")
