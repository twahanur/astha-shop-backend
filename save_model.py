# save_model.py
import tensorflow
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import GlobalMaxPooling2D

print("Loading and saving Keras model...")
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
feature_extractor_model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
# Save the model in the Keras format
feature_extractor_model.save('feature_extractor_model.keras')
print("✅ Model saved as feature_extractor_model.keras")