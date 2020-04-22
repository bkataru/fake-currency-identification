import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def sub(og, name):
    h, w, c = og.shape
    x_start_r = 0.3125  # 200
    x_end_r = 0.703125  # 450
    y_start_r = 0.270833  # 130
    y_end_r = 0.791666  # 380

    x = int(np.floor(x_start_r * w))
    y = int(np.floor(y_start_r * h))
    x2 = int(np.floor(x_end_r * w))
    y2 = int(np.floor(y_end_r * h))

    image = og[y:y2, x:x2].copy()

    RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    edged = cv2.Canny(RGB, 140, 100)
    return edged

directory = '.\\bounding-test'
if directory == '.':
    files = ['test.jpg', 'test1.jpg', 'test2.jpg', 'test3.jpg']
    for filename in files:
        img = cv2.imread(directory + '\\' + filename)
        img = sub(img, filename)
        cv2.imshow("image", img)
        cv2.waitKey(0)
else:
    files = os.listdir(directory)
    for filename in files:
        img = cv2.imread(directory + '\\' + filename)
        img = sub(img, filename)
        print(img.shape)
        cv2.imshow("image", img)
        cv2.waitKey(0)

    for filename in files:
        img = cv2.imread(directory + '\\' + filename)
        img = cv2.rotate(img, cv2.ROTATE_180)
        img = sub(img, filename)
        cv2.imshow("image", img)
        cv2.waitKey(0)