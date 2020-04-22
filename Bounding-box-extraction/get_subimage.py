import cv2
import numpy as np
import os

def sub(og):
    h, w, c = og.shape
    x_start_r = 0.3125  # 200
    x_end_r = 0.703125  # 450
    y_start_r = 0.270833  # 130
    y_end_r = 0.791666  # 380

    x = int(np.floor(x_start_r * w))
    y = int(np.floor(y_start_r * h))
    x2 = int(np.floor(x_end_r * w))
    y2 = int(np.floor(y_end_r * h))

    image = og.copy() # og[y:y2, x:x2].copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)

    return thresh

directory = '.\\bounding-test'
if directory == '.':
    files = ['test.jpg', 'test1.jpg', 'test2.jpg', 'test3.jpg']
    for filename in files:
        img = cv2.imread(directory + '\\' + filename)
        img = sub(img)
        cv2.imshow("image", img)
        cv2.waitKey(0)
else:
    files = os.listdir(directory)
    for filename in files:
        img = cv2.imread(directory + '\\' + filename)
        img = sub(img)
        print(img.shape)
        cv2.imshow("image", img)
        cv2.waitKey(0)

    for filename in files:
        img = cv2.imread(directory + '\\' + filename)
        img = cv2.rotate(img, cv2.ROTATE_180)
        img = sub(img)
        cv2.imshow("image", img)
        cv2.waitKey(0)