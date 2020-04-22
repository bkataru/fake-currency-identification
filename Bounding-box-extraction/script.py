import numpy as np
import cv2

original_image = cv2.imread("test2.jpg")
image = original_image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
thresh = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilate = cv2.dilate(thresh, kernel, iterations = 1)

cv2.imshow("thresh", thresh)
cv2.imshow("dilate", dilate)

# Find contours in the image
cnts = cv2.findContours(dilate.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

contours = []

threshold_min_area = 400
threshold_max_area = 3000

for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    if area > threshold_min_area and area < threshold_max_area:
        # cv2.drawContours(original_image,[c], 0, (0,255,0), 3)
        cv2.rectangle(original_image, (x,y), (x+w, y+h), (0,255,0),1)
        px_area = w * h

        # if px_area in range(3320 - 50, 3320 + 50):
        #     sub = original_image[y: y+h, x: x+w].copy()
        #     cv2.imshow("{}".format(x + y + x * y), sub)
        #     cv2.waitKey(0)

        print(px_area)
        # sub = original_image[y: y + h, x: x + w].copy()
        # cv2.imshow("{}".format(x + y + x * y), sub)
        # cv2.waitKey(0)

        print((x, x+w), (y, y+h))
        contours.append(c)

cv2.imshow("detected", original_image)
print('contours detected: {}'.format(len(contours)))
cv2.waitKey(0)