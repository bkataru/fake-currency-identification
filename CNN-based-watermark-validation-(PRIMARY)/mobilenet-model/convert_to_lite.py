import tensorflow as tf

model = tf.keras.models.load_model('watermark_mobilenetmodel.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("converted_watermark_mobilenetmodel.tflite", "wb").write(tflite_model)

