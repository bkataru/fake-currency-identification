from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.models import Model
from keras.applications import MobileNet, MobileNetV2

model1 = MobileNet(weights='imagenet', input_shape= (224, 224, 3))
model2 = MobileNet(weights='imagenet', input_shape= (224, 224, 3), include_top=False)

model3 = MobileNetV2(weights='imagenet', input_shape= (224, 224, 3))

print(len(model1.layers))
print(len(model3.layers))

