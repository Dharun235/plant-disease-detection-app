import tensorflow as tf

# Load the model (replace with your own model path)
model = tf.keras.models.load_model('plant_disease_model.h5')

# Convert to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
