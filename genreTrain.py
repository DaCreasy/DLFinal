from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
from keras import backend as K

w,h = 678, 512
train_dir = 'theSpects/train'
test_dir = 'theSpects/testing'

num_train_samples = 800
num_test_samples = 200
eps = 50
b_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, h, w)
else:
    input_shape = (h, w, 3)

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print('Model has been compiled')

train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(h,w),
    batch_size=b_size,
    class_mode='categorical')

print('train_gen created')

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(h,w),
    batch_size=b_size,
    class_mode='categorical')

print('test_gen created')

model.fit_generator(
    train_gen,
    steps_per_epoch=num_train_samples // b_size,
    epochs=eps,
    validation_data=test_gen,
    validation_steps=num_test_samples // b_size)

model.save_weights('first_run.h5')
