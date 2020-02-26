import math
import keras
from keras import backend as K
from keras import regularizers
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam, RMSprop, SGD, Adadelta
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.applications import MobileNetV2
from keras.applications.mobilenet import preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt

batch_size = 16

train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
training_set = train_gen.flow_from_directory(
    directory="../dataset/training_set",
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_set = test_gen.flow_from_directory(
    directory="../dataset/test_set",
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

#imports the mobilenet model and discards the last fully connected neuron layer.
mobilenetmodel = MobileNetV2(weights='imagenet', input_shape= (224, 224, 3), include_top=False)
print(mobilenetmodel.summary())

x = mobilenetmodel.output
x = GlobalAveragePooling2D()(x)
preds = Dense(3, activation='softmax', kernel_regularizer=regularizers.l2(0.001))(x) # final layer with softmax activation

model_final = Model(inputs = mobilenetmodel.input, outputs = preds)

training_size = 6000
validation_size = 3000
steps_per_epoch = math.ceil(training_size / batch_size)
validation_steps = math.ceil(validation_size / batch_size)

# compilation 1
optimizer1 = SGD(lr=0.001)
model_final.compile(optimizer = optimizer1, loss='categorical_crossentropy',metrics=['accuracy'])
print("total layer count", len(model_final.layers))

print(model_final.summary())
earlystop1 = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
hist1 = model_final.fit_generator(
    training_set,
    steps_per_epoch=steps_per_epoch,
    epochs=20,
    validation_data = test_set,
    validation_steps=validation_steps,
    callbacks=[earlystop1],
    workers=10,
    shuffle=True
)

model_final.save("uv_mobilenetmodel.h5")

print("mobilenet_uv_class_indices", training_set.class_indices)
f = open("mobilenet_uv_class_indices.txt", "w")
f.write(str(training_set.class_indices))
f.close()

plt.plot(hist1.history["accuracy"])
plt.plot(hist1.history['val_accuracy'])
plt.plot(hist1.history['loss'])
plt.plot(hist1.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.savefig('mobilenet' + '_plot.png')
plt.show()