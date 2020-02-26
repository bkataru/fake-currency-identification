import tensorflow as tf

model = tf.keras.models.load_model('uv_vgg16model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("converted_uv_vgg16model.tflite", "wb").write(tflite_model)

