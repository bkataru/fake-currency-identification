import math
import matplotlib.pyplot as plt
import keras
from keras import regularizers
from keras.models import Model
from keras.layers import Dense, Flatten, BatchNormalization, Dropout
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, EarlyStopping

batch_size = 32

train_gen = ImageDataGenerator()
training_set = train_gen.flow_from_directory(
    directory="../dataset/training_set",
    target_size=(224,224),
    batch_size=batch_size
)

test_gen = ImageDataGenerator()
test_set = test_gen.flow_from_directory(
    directory="../dataset/test_set",
    target_size=(224,224),
    batch_size=batch_size
)


vggmodel = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

print(vggmodel.summary())

for layers in (vggmodel.layers):
    layers.trainable = False

x = Flatten()(vggmodel.output)

x = Dense(4096, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.7)(x)

x = Dense(4096, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.7)(x)

predictions = Dense(2, activation='softmax', kernel_initializer='random_uniform',
                    bias_initializer='random_uniform', bias_regularizer=regularizers.l2(0.01), name='predictions')(x)

model_final = Model(input = vggmodel.input, output = predictions)

training_size = 4000
validation_size = 2000
steps_per_epoch = math.ceil(training_size / batch_size)
validation_steps = math.ceil(validation_size / batch_size)

# compilation 1
rms = optimizers.RMSprop(lr=0.0001, decay=1e-4)
model_final.compile(loss="categorical_crossentropy", optimizer = rms, metrics=["accuracy"])
print(model_final.summary())
earlystop1 = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
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

plt.plot(hist1.history["accuracy"])
plt.plot(hist1.history['val_accuracy'])
plt.plot(hist1.history['loss'])
plt.plot(hist1.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.savefig('vgg16' + '_initialModel_plot.png')

# make last 8 layers trainable

for layer in model_final.layers[:15]:
    layer.trainable = False

for layer in model_final.layers[15:]:
    layer.trainable = True

# compilation 2
sgd = optimizers.SGD(lr=0.0001, decay=1e-4, momentum=0.9, nesterov=True)
model_final.compile(loss = "categorical_crossentropy", optimizer = sgd, metrics=["accuracy"])

earlystop2 = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
hist2 = model_final.fit_generator(
    training_set,
    epochs= 20,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    validation_data= test_set,
    callbacks=[earlystop2],
    workers=10,
    shuffle=True
)

model_final.save("watermark_vgg16model.h5")

print("vgg16_watermark_class_indices", training_set.class_indices)
f = open("vgg16_watermark_class_indices.txt", "w")
f.write(str(training_set.class_indices))
f.close()

plt.plot(hist2.history["accuracy"])
plt.plot(hist2.history['val_accuracy'])
plt.plot(hist2.history['loss'])
plt.plot(hist2.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.savefig('vgg16' + '_finalModel_plot.png')
plt.show()


