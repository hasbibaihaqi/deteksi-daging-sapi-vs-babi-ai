import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os

# Create a small dummy model with the expected architecture output shape
input_tensor = Input(shape=(299, 299, 3))
x = GlobalAveragePooling2D()(input_tensor)
x = Dense(256, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_tensor, outputs=predictions)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Ensure models directory exists
os.makedirs('models', exist_ok=True)
model.save('models/meat_model.h5')

print("Mock model saved to models/meat_model.h5")
