import cv2
import random
from skimage import transform as tf
import numpy as np
from scipy.ndimage import rotate

def cv2_clipped_zoom(img, zoom_factor):
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of
    the image without changing dimensions
    Args:
        img : Image array
        zoom_factor : amount of zoom as a ratio (0 to Inf)
    """
    height, width = img.shape[:2] # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result

#image = cv2.imread('10back1-pr.jpg')

for i in range(1, 51):
    contrast = random.randint(70, 130)/100 # 0.7 to 1.3
    brightness = random.randint(-40, 40) # 40 to -40
    shear = random.randint(-20, 20)/100 # 0.2 to -0.2
    zoom = random.randint(100, 130)/100 # 1 to 1.3
    rot = random.randint(-100, 100)/10 # -10 to 10

    image = cv2.imread('10back1-pr.jpg')
    image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

    affine_tf = tf.AffineTransform(shear=shear)
    image = tf.warp(image, inverse_map=affine_tf)
    image = cv2_clipped_zoom(image, zoom)
    image = rotate(image, rot, reshape=False)

    image = cv2.convertScaleAbs(image, alpha=(255.0))
    cv2.imwrite('.\\copies\\{}.jpg'.format(i), image)