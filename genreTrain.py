''''
This code is based on the Keras example at https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d
'''

from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, \
    BatchNormalization, Conv2D, MaxPooling2D
from keras.optimizers import Adam
import numpy as np
from keras import backend as K

w,h = 678, 512
train_dir = 'data/train'
test_dir = 'data/test'

print(f"Image size: {w}x{h}")
print(f"Train directory: {train_dir}")
print(f"Test directory: {test_dir}")

num_train_samples = 600
num_test_samples = 200
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
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

opt = Adam(lr=0.001)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

print('Model has been compiled')

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


model.fit_generator(
    train_gen,
    steps_per_epoch=num_train_samples // b_size,
    epochs=eps,
    validation_data=test_gen,
    validation_steps=num_test_samples // b_size,
    callbacks=[
        ReduceLROnPlateau(
            monitor='val_acc', factor=0.5, patience=10, min_delta=0.01)])

saveFile = 'first_run.h5'

model.save_weights(saveFile)
print(f"Saving model to {saveFile}")
