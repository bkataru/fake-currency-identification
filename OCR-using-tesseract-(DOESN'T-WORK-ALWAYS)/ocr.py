from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=26, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened



image = cv.imread('hc4.jpg')
image = cv.cvtColor(image, cv.IMREAD_GRAYSCALE)
sharpened_image = unsharp_mask(image)
plot_image = np.concatenate((image, sharpened_image), axis=1)
(plt.imshow(plot_image), plt.show())

confs = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
for i in confs:
    amounts = []
    for k in range(1, 50):
        sharpened_image = unsharp_mask(image, amount=float(k))
        text = pytesseract.image_to_string(sharpened_image, lang='eng', config='--psm ' + str(i))
        if "20\n" in text:
            amounts.append(k)
    print("psm mode: ", i)
    print("configs", amounts)
    print("=" * 50)

text = pytesseract.image_to_string(sharpened_image, lang='eng', config='--psm 11 --oem 3')
print(text)