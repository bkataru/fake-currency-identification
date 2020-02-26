import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model
import ast, os

model = load_model('currency_mobilenetmodel.h5')
f = open("mobilenet_currency_class_indices.txt", "r")
labels = f.read()
labels = ast.literal_eval(labels)
final_labels = {v: k for k, v in labels.items()}


def predict_image(imgname, from_test_dir):
    test_image = image.load_img(imgname, target_size = (224, 224))

    # plt.imshow(test_image)
    # plt.show()

    test_image = np.asarray(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = (2.0 / 255.0) * test_image - 1.0
    result = model.predict(test_image)

    result_dict = dict()
    for key in list(final_labels.keys()):
        result_dict[final_labels[key]] = result[0][key]
    sorted_results = {k: v for k, v in sorted(result_dict.items(), key=lambda item: item[1], reverse=True)}

    if not from_test_dir:
        print('=' * 50)
        for label in sorted_results.keys():
            print("{}: {}%".format(label, sorted_results[label] * 100))

    final_result = dict()
    final_result[list(sorted_results.keys())[0]] = sorted_results[list(sorted_results.keys())[0]] * 100

    return final_result

def verify_test_dir():
    path = '..\\batch-test-images'
    files = os.listdir(path)
    file_count = len(files)
    finals = []
    for filename in files:

        digits = []
        for al in filename:
            if al.isdigit():
                digits += al
            else:
                break

        num = "".join(digits)

        final_string = []
        final_string.append(num)

        if 'old' in filename:
            final_string.append('old')
        else:
            final_string.append('new')

        if 'back' in filename:
            final_string.append('back')
        elif 'front' in filename:
            final_string.append('front')

        rev = filename.split('.')[0][-1:]
        if rev == 'r':
            final_string.append('down')
        else:
            final_string.append('up')

        final_string = "_".join(final_string)

        prediction = predict_image(path + '\\' + filename, True)
        if list(prediction.keys())[0] == final_string:
            print("{}: Correct Prediction".format(filename))
        else:
            print("{}: INCORRECT PREDICTION!".format(filename))


print('=' * 50)
final_result = predict_image('..\\test-images\\kappa.jpg', False)
print("Final Result: ", final_result)
# verify_test_dir()




