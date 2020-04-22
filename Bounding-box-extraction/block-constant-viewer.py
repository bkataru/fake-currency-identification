import numpy as np
import cv2

for block in range(3, 30, 2): # > 0
    for constant in range(-4, 5):
        files = ['test.jpg']

        for filename in files:
            original_image = cv2.imread(filename)
            image = original_image.copy()

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, constant)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilate = cv2.dilate(thresh, kernel, iterations = 1)


            # Find contours in the image
            cnts = cv2.findContours(dilate.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]

            contours = []

            threshold_min_area = 400
            threshold_max_area = 3000

            final_copy = dilate.copy()
            for c in cnts:
                x,y,w,h = cv2.boundingRect(c)
                area = cv2.contourArea(c)
                if area > threshold_min_area and area < threshold_max_area:
                    # cv2.drawContours(original_image,[c], 0, (0,255,0), 3)
                    cv2.rectangle(final_copy, (x,y), (x+w, y+h), (0,255,0),1)
                    px_area = w * h

                    # if px_area in range(3320 - 50, 3320 + 50):
                    #     sub = original_image[y: y+h, x: x+w].copy()
                    #     cv2.imshow("{}".format(x + y + x * y), sub)
                    #     cv2.waitKey(0)

                    # sub = original_image[y: y + h, x: x + w].copy()
                    # cv2.imshow("{}".format(x + y + x * y), sub)
                    # cv2.waitKey(0)

                    contours.append(c)

            cv2.imshow("{}/{}/{}".format(filename.split('.')[0], block, constant), final_copy)
            cv2.waitKey(0)

        cv2.destroyAllWindows()

