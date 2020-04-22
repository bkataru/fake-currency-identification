import cv2
import numpy as np
import os

def draw_rec(img):
    h, w, c = img.shape
    x_start_r = 0.3125 # 200
    x_end_r = 0.703125 # 450
    y_start_r = 0.270833 # 130
    y_end_r = 0.791666 # 380

    greens_left = []
    for i in range(1, 25):
        greens_left.append(img[int(np.floor(h / 2))][i][1])

    greens_right = []
    for i in range(1, 25):
        greens_right.append(img[int(np.floor(h / 2))][w - i][1])

    if np.median(greens_left) > 100:
        left_out = True
    else:
        left_out = False

    if np.median(greens_right) > 100:
        right_out = True
    else:
        right_out = False

    x_adjust = 0
    # if not (right_out and left_out):
    #     if not right_out and left_out:
    #         x_adjust = 50
    #     elif right_out and not left_out:
    #         x_adjust = -50

    x = int(np.floor(x_start_r * w)) + x_adjust
    y = int(np.floor(y_start_r * h))
    x2= int(np.floor(x_end_r * w)) + x_adjust
    y2 = int(np.floor(y_end_r * h))

    img = cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 1)
    return img

directory = '.\\bounding-test'
if directory == '.':
    files = ['test.jpg', 'test1.jpg', 'test2.jpg', 'test3.jpg']
    for filename in files:
        img = cv2.imread(directory + '\\' + filename)
        img = draw_rec(img)
        cv2.imshow("image", img)
        cv2.waitKey(0)
else:
    files = os.listdir(directory)
    for filename in files:
        img = cv2.imread(directory + '\\' + filename)
        img = draw_rec(img)
        cv2.imshow("image", img)
        cv2.waitKey(0)

    for filename in files:
        img = cv2.imread(directory + '\\' + filename)
        img = cv2.flip(img, 1)
        img = draw_rec(img)
        cv2.imshow("image", img)
        cv2.waitKey(0)