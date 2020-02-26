import cv2
import os, random
from skimage import transform as tf
import numpy as np
from scipy.ndimage import rotate
import progressbar

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

def modify_image(img_name, inverted_flag, ot_flag):
    contrast = random.randint(70, 130) / 100  # 0.7 to 1.3
    brightness = random.randint(-40, 40)  # 40 to -40
    shear = random.randint(-20, 20) / 100  # 0.2 to -0.2
    zoom = random.randint(100, 130) / 100  # 1 to 1.3
    rot = random.randint(-100, 100) / 10  # -10 to 10

    if ot_flag:
        zoom = zoom - 0.4
        brightness = random.randint(-40, 20)

    image = cv2.imread(img_name)

    if inverted_flag:
        image = rotate(image, 180, reshape=False)
    image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

    affine_tf = tf.AffineTransform(shear=shear)
    image = tf.warp(image, inverse_map=affine_tf)
    image = cv2_clipped_zoom(image, zoom)
    image = rotate(image, rot, reshape=False)

    image = cv2.convertScaleAbs(image, alpha=(255.0))
    return image

folders = []
for name in os.listdir('.\\comp_data'):
    if len(name.split('.')) == 1:
        folders.append(name)
print(folders)

global_count = 0
training_pr_count = 1200
training_ot_count = 300
test_pr_count = 600
test_ot_count = 150

# 2 * (1200 + 300) * 2 + 2 * (600 + 150) * 2
total = len(folders) * (training_pr_count + training_ot_count) * 2 + len(folders) * (test_pr_count + test_ot_count) * 2
print(total)
total += 500
bar = progressbar.ProgressBar(max_value=total)
def main(dataset_type):
    for fold in folders:
        files = []
        for temp in os.listdir('.\\comp_data\\' + fold):
            if not ('.' in temp and temp.split('.')[1] == 'jpg'):
                raise NameError('Not all files are images in folder {}'.format(fold))
            else:
                files.append(temp)

        pr_list = []
        ot_list = []
        for name in files:
            if '-' in name and name.split('-')[1] == 'pr.jpg':
                pr_list.append(name)
            else:
                ot_list.append(name)

        def generate_images(img_list, num, id, inverted_flag):
            global global_count
            UNIQUE = 1

            ot_flag = False
            if id == 'OT':
                ot_flag = True

            for img_ind in range(0, len(img_list)):
                iterations = num // len(img_list)
                rem = 0
                if num % len(img_list) != 0:
                    rem = num % len(img_list)
                print(rem)

                for iter in range(1, iterations + 1):
                    image = modify_image('.\\comp_data\\' + fold + '\\' + img_list[img_ind], inverted_flag, ot_flag)
                    if inverted_flag:
                        cv2.imwrite('.\\{}\\{}\\{}-{}-{}-inv.jpg'.format(dataset_type, fold, id, iter, UNIQUE), image)
                    else:
                        cv2.imwrite('.\\{}\\{}\\{}-{}-{}.jpg'.format(dataset_type, fold, id, iter, UNIQUE), image)

                    global_count += 1
                    print("Percentage done: {}%".format(round((global_count/total) * 100, 3)))
                    bar.update(global_count)
                if rem != 0:
                    for rem_iter in range(1, rem + 1):
                        image = modify_image('.\\comp_data\\' + fold + '\\' + img_list[0], inverted_flag, ot_flag)
                        if inverted_flag:
                            cv2.imwrite('.\\{}\\{}\\{}-{}-rem-inv.jpg'.format(dataset_type, fold, id, rem_iter), image)
                        else:
                            cv2.imwrite('.\\{}\\{}\\{}-{}-rem.jpg'.format(dataset_type, fold, id, rem_iter), image)

                    global_count += 1
                    print("Percentage done: {}%".format(round((global_count/total) * 100, 3)))
                    bar.update(global_count)

                UNIQUE += 1

        if dataset_type == 'training_set':
            generate_images(pr_list, training_pr_count, "PR", False)
            generate_images(ot_list, training_ot_count, "OT", False)
            generate_images(pr_list, training_pr_count, "PR", True)
            generate_images(ot_list, training_ot_count, "OT", True)
        elif dataset_type == 'test_set':
            generate_images(pr_list, test_pr_count, "PR", False)
            generate_images(ot_list, test_ot_count, "OT", False)
            generate_images(pr_list, test_pr_count, "PR", True)
            generate_images(ot_list, test_ot_count, "OT", True)

main('training_set')
main('test_set')







