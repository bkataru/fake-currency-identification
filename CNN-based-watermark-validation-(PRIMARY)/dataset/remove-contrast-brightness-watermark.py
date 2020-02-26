import os
from PIL import Image
import matplotlib.pyplot as plt

for img_name in os.listdir():
    if len(img_name.split('.')) > 1:
        if img_name.split('.')[1] == 'jpg':
            img = Image.open(img_name)
            for h in range(0, 61):
                for w in range(0, 221):
                    img.putpixel((w, h), (0, 0, 0))

            img.save(img_name)