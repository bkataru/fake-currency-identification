import matplotlib.pyplot as plt
from keras.preprocessing import image
from PIL import Image
import numpy as np

img = image.load_img('10front.jpg', target_size=(224, 224))
plt.imshow(img)
img = np.asarray(img)
plt.imshow(img)
plt.show()
