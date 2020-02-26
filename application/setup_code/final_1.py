from tflite_runtime.interpreter import Interpreter
from PIL import Image
import numpy as np
import os, ast, re
import cv2

def get_labels(path):	
	f = open(path, "r")
	labels = f.read()
	labels = ast.literal_eval(labels)

	final_labels = {v: k for k, v in labels.items()}
	
	return final_labels
  
def predict_image(imgname, final_labels, interpreter, input_details, output_details):
    test_image = cv2.imread(imgname)
    test_image = cv2.resize(test_image, (224, 224), cv2.INTER_AREA)
    cv2.imshow('kappa', test_image)
    cv2.waitKey(0)
    test_image = np.expand_dims(test_image, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], test_image)
    interpreter.invoke()
    result = interpreter.get_tensor(output_details[0]['index'])

    result_dict = dict()
    for key in list(final_labels.keys()):
        result_dict[final_labels[key]] = result[0][key]
    sorted_results = {k: v for k, v in sorted(result_dict.items(), key=lambda item: item[1], reverse=True)}

    for label in sorted_results.keys():
        print("{}: {}%".format(label, sorted_results[label] * 100))

    final_result = dict()
    final_result[list(sorted_results.keys())[0]] = sorted_results[list(sorted_results.keys())[0]] * 100

    return final_result


classify_interpreter = Interpreter(model_path="converted_currency_model.tflite")
watermark_interpreter = Interpreter(model_path="converted_watermark_model.tflite")
uv_interpreter = Interpreter(model_path="converted_uv_model.tflite")

classify_input_details = classify_interpreter.get_input_details()
classify_output_details = classify_interpreter.get_output_details()
watermark_input_details = watermark_interpreter.get_input_details()
watermark_output_details = watermark_interpreter.get_output_details()
uv_input_details = uv_interpreter.get_input_details()
uv_output_details = uv_interpreter.get_output_details()

classify_interpreter.allocate_tensors()
watermark_interpreter.allocate_tensors()
uv_interpreter.allocate_tensors()

currency_labels = get_labels('currency_class_indices.txt')
uv_labels = get_labels('uv_class_indices.txt')
watermark_labels = get_labels('watermark_class_indices.txt')

res = predict_image('test.jpg', currency_labels, classify_interpreter, classify_input_details, classify_output_details)
print("=" * 50)
print(res)

