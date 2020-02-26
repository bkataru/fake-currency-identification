import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import subprocess, os

import utils
import subprocess, os

orb = cv2.ORB_create()
# orb is an alternative to SIFT

test_img = utils.read_img('test1.jpg')

#resizing must be dynamic
test_img = utils.resize_img(test_img, 0.2)
# display('original', original)

# keypoints and descriptors
# (kp1, des1) = orb.detectAndCompute(test_img, None)
(kp1, des1) = orb.detectAndCompute(test_img, None)

training_set = os.listdir('training-data/')
for ind in range(0, len(training_set)):
    training_set[ind] = 'training-data/' + training_set[ind]

max_val = 8
max_pt = -1
max_kp = 0
good = []

for i in range(0, len(training_set)):
    # train image
    train_img = cv2.imread(training_set[i])

    (kp2, des2) = orb.detectAndCompute(train_img, None)

    # brute force matcher
    bf = cv2.BFMatcher()

    all_matches = bf.knnMatch(des1, des2, k=2)

    # give an arbitrary number -> 0.789
    # if good -> append to list of good matches
    for (m, n) in all_matches:
        if m.distance < 0.789 * n.distance:
            good.append([m])

    if len(good) > max_val:
        max_val = len(good)
        max_pt = i
        max_kp = kp2

    print(i, ' ', training_set[i], ' ', len(good))

if max_val != 8:
    print(training_set[max_pt])
    print('good matches ', max_val)

    train_img = cv2.imread(training_set[max_pt])
    img3 = cv2.drawMatchesKnn(test_img, kp1, train_img, max_kp, good, 4)

    note = str(training_set[max_pt]).split('_')[0].split('/')[1]
    print('\nDetected denomination: Rs. ', note)

    (plt.imshow(img3), plt.show())
else:
    print('No Matches')