''''
This code is based on the Keras example at https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d
'''

from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, \
    BatchNormalization, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras import backend as K
from ann_visualizer.visualize import ann_viz
import numpy as np
import matplotlib.pyplot as plt

w,h = 678, 512
train_dir = 'data/train'
test_dir = 'data/test'

print(f"Image size: {w}x{h}")
print(f"Train directory: {train_dir}")
print(f"Test directory: {test_dir}")

num_train_samples = 800
num_test_samples = 100
eps = 1000
b_size = 16

print(f"Epochs: {eps}")
print(f"Batch size: {b_size}")

if K.image_data_format() == 'channels_first':
    input_shape = (3, h, w)
else:
    input_shape = (h, w, 3)

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = input_shape))
model.add(Activation('softplus'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(32, (3,3)))
model.add(Activation('softplus'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(32, (3,3)))
model.add(Activation('softplus'))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('softplus'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

opt = Adam(lr=0.001)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['categorical_accuracy'])

print('Model has been compiled')
#'''A
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(h,w),
    batch_size=b_size,
    class_mode='categorical')

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(h,w),
    batch_size=b_size,
    class_mode='categorical')


history = model.fit_generator(
    train_gen,
    steps_per_epoch=num_train_samples // b_size,
    epochs=eps,
    validation_data=test_gen,
    validation_steps=num_test_samples // b_size,
    verbose=2,
    callbacks=[
        ReduceLROnPlateau(
            monitor='val_acc', factor=0.5, patience=10, min_delta=0.01)])

#A'''

'''B
modelArch = "5_layer_cnn.gv"
archTitle = "Five Layer CNN"
print(f"Saving model architecture to {modelArch}")
ann_viz(model, title=archTitle, filename=modelArch)
B'''

curRun = "run_2"

#'''C
accFile = "images/"  + curRun + "/acc.png"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(accFile, bbox_inches='tight')

lossFile = "images/" + curRun + "/loss.png"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(lossFile, bbox_inches='tight')

modelResults = 'models/second_run.h5'

model.save_weights(modelResults)
print(f"Saving model results to {modelResults}")
#C'''
