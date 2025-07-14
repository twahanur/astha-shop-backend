import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import GlobalMaxPooling2D

print("Loading original Keras model...")
# Recreate the exact same model structure you had before
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
feature_extractor_model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

print("Converting model to TensorFlow Lite...")
# Create a TFLite converter from the Keras model
converter = tf.lite.TFLiteConverter.from_keras_model(feature_extractor_model)

# Optional: Apply optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model to a file
with open('feature_extractor.tflite', 'wb') as f:
    f.write(tflite_model)

print("Successfully converted and saved as 'feature_extractor.tflite'")