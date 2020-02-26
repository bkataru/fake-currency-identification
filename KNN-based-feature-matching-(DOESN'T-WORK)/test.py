import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import subprocess, os

import utils
import subprocess, os
orb = cv2.ORB_create()
# sift = cv2.xfeatures2d.SIFT_create()
# orb is an alternative to SIFT

img1 = utils.read_img('test.jpg')
img1 = utils.resize_img(img1, 0.2)

# keypoints and descriptors
(kp1, des1) = orb.detectAndCompute(img1, None)

training_set = os.listdir('training-data/')
for ind in range(0, len(training_set)):
    training_set[ind] = 'training-data/' + training_set[ind]

matched = []
max_kp = 0
max_ind = -1
MIN_MATCH_COUNT = 10

# FLANN parameters
FLANN_INDEX_LSH = 0
index_params= dict(algorithm = 6,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
search_params = dict(checks=100)   # or pass empty dictionary


for i in range(0, len(training_set)):
    img2 = cv2.imread(training_set[i])
    (kp2, des2) = orb.detectAndCompute(img2, None)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # ratio test as per Lowe's paper
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    plt.imshow(img3, 'gray'), plt.show()

#     matched.append((i, len(matches), matches, kp2, len(kp2)))
#     print(training_set[i], len(kp2))
#
# matched = sorted(matched, key = lambda x: x[4])
# train_img = cv2.imread(training_set[max_ind])
# img3 = cv2.drawMatches(img1, kp1, train_img, matched[len(matched) - 1][3], matched[len(matched) - 1][2], None, flags=2)
# plt.imshow(img3)
# plt.show()