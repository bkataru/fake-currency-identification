import tensorflow as tf
from PIL import Image
import numpy as np
import os, ast
import cv2

interpreter = tf.lite.Interpreter(model_path="converted_watermark_vgg16model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("input_details", input_details)
print("output_details", output_details)

f = open("vgg16_watermark_class_indices.txt", "r")
labels = f.read()
labels = ast.literal_eval(labels)

final_labels = {v: k for k, v in labels.items()}

print(final_labels)

def predict_image(imgname, from_test_dir):
    test_image = cv2.imread(imgname)
    test_image = cv2.resize(test_image, (224, 224), cv2.INTER_AREA)
    test_image = np.expand_dims(test_image, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], test_image)
    interpreter.invoke()
    result = interpreter.get_tensor(output_details[0]['index'])
    #print(result)

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
    folders = os.listdir(path)

    correct_preds = 0
    file_count = 0
    for fold in folders:
        files = os.listdir(path + '\\' + fold)
        for filename in files:
            final_string = fold
            prediction = predict_image(path + '\\{}\\'.format(fold) + filename, True)
            if list(prediction.keys())[0] == final_string:
                print("{}\{}: Correct Prediction".format(fold, filename))
                correct_preds += 1
            else:
                print("{}\{}: INCORRECT PREDICTION".format(fold, filename))
            file_count += 1

    print(correct_preds, file_count)

print('=' * 50)
print(predict_image('..\\test-images\\yesw.jpg', False))
# verify_test_dir()
