import tensorflow as tf
from PIL import Image
import PIL
import numpy as np
import os, ast
import cv2

interpreter = tf.lite.Interpreter(model_path="converted_currency_mobilenetmodel.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("input_details", input_details)
print("output_details", output_details)

f = open("mobilenet_currency_class_indices.txt", "r")
labels = f.read()
labels = ast.literal_eval(labels)

final_labels = {v: k for k, v in labels.items()}

def predict_image(imgname, from_test_dir):
    test_image = cv2.imread(imgname)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    test_image = cv2.resize(test_image, (224, 224), cv2.INTER_AREA)
    test_image = np.expand_dims(test_image, axis=0)

    # Normalize from [-1, 1]
    test_image = (2.0 / 255.0) * test_image - 1.0
    test_image = test_image.astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], test_image)
    interpreter.invoke()
    result = interpreter.get_tensor(output_details[0]['index'])

    result_dict = dict()
    for key in list(final_labels.keys()):
        result_dict[final_labels[key]] = result[0][key]
    sorted_results = {k: v for k, v in sorted(result_dict.items(), key=lambda item: item[1], reverse=True)}

    if not from_test_dir:
        for label in sorted_results.keys():
            print("{}: {}%".format(label, sorted_results[label] * 100))

    final_result = dict()
    final_result[list(sorted_results.keys())[0]] = sorted_results[list(sorted_results.keys())[0]] * 100

    return final_result

def verify_test_dir():
    path = '..\\batch-test-images'
    files = os.listdir(path)

    correct_preds = 0
    file_count = 0

    finals = []
    for filename in files:
        final_string = []

        denom = ''
        for char in filename:
            if char.isdigit():
                denom += char
            else:
                break
        final_string.append(denom)

        if 'old' in filename:
            final_string.append('old')
        else:
            final_string.append('new')

        if 'back' in filename:
            final_string.append('back')
        else:
            final_string.append('front')

        if filename.split('.')[0][-1] == 'r':
            final_string.append('down')
        else:
            final_string.append('up')

        final_string = "_".join(final_string)

        prediction = predict_image(path + '\\' + filename, True)
        if list(prediction.keys())[0] == final_string:
            print("{}: Correct Prediction".format(filename), prediction)
            correct_preds += 1
        else:
            print("{}: INCORRECT PREDICTION".format(filename), prediction)
        file_count += 1

    print(correct_preds, file_count)

print('=' * 50)
# print(predict_image('..\\test-images\\kappa.jpg', False))
verify_test_dir()
