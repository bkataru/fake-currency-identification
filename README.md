# Fake Currency Identification

## Abstract
Uses Deep Learning Neural Networks to determine whether an Indian currency note is fake or real.

## Technical
Uses retrained MobileNetV2 image classification models to determine whether an Indian currency note is fake or real. The process happens in three steps:
1. **Currency Classification Model**: Identifies the denomination, type (old or new), face orientation (front or back), and alignment (up or down) from a frontlight image of the note.
2. **Watermark Identification Model**: Identifies whether the note contains the Gandhi watermark under backlight illumination from a backlight image of the note.
3. **Ultraviolet Strip Detection Model**: Identifies whether the note contains a fluorescent strip or not and if yes then whether the strip is continuous or dashed under Ultraviolet illumination from a ultraviolet image of the note.

The models were meant to be integrated together to run on a Raspberry Pi (hence the choice of MobileNet for the architecture) but you can strip and use each model individually if you like.
### A Bit of History
Initially, I tried using KNN-based ORB feature matching to compare features of currency images and detect currency denominations but that ended up being highly inaccurate. I later tried running OCR using Tesseract in hopes of detecting the denomination which sort of worked but more often than not didn't. Eventually, I settled on using Convolutional Neural Networks for my task.

## Getting Started
Explore around the folders in the project and check out the code. I've placed text files at appropriate places/folders to act as instructions that will guide you in case you want to do any sort of training yourself. This repo has all the code files you need for retraining, transfer learning, or testing any of my models.

## Example usage

### MobileNetV2

```console
foo@bar:~$ cd CNN-based-classification-(PRIMARY)
foo@bar:~$ cd mobilenet-model

# Training a new classification model from scratch using data located in ./dataset/
foo@bar:~$ python3 retrained_mobilenet_train.py

# Testing a pretrained model
foo@bar:~$ python3 retrained_mobilenet_test.py

# Testing a lite version of the same model
foo@bar:~$ python3 lite_run.py

# Converting a Keras .h5 model to tflite
foo@bar:~$ python3 convert_to_lite.py

```

---

### VGG16

```console
foo@bar:~$ CNN-based-classification-(PRIMARY)
foo@bar:~$ vgg16-model

# Training a new classification model from scratch using data located in ./dataset/
foo@bar:~$ python3 pretrained_vgg16_train.py

# Running a pretrained model
foo@bar:~$ python3 pretrained_vgg16_test.py

# Running a lite version of the same model
foo@bar:~$ python3 lite_run.py

# Converting a Keras .h5 model to tflite
foo@bar:~$ python3 convert_to_lite.py
```