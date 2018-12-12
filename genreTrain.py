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

# Width and height of images
w,h = 678, 512
train_dir = 'data/train'
test_dir = 'data/test'

print(f"Image size: {w}x{h}")
print(f"Train directory: {train_dir}")
print(f"Test directory: {test_dir}")

# Number of data points, epochs and batch size
num_train_samples = 800
num_test_samples = 100
eps = 1000
b_size = 16

print(f"Epochs: {eps}")
print(f"Batch size: {b_size}")

# Format of numpy arrays is either channels first or channels last
if K.image_data_format() == 'channels_first':
    input_shape = (3, h, w)
else:
    input_shape = (h, w, 3)
    
print('Building model architecture')
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
#'''A
#Use data generators to get the images, apply preprocessing, and generate numpy arrays
#Train the model
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
    callbacks=[
        ReduceLROnPlateau(
            monitor='val_categorical_accuracy', factor=0.5, patience=10, min_delta=0.01)])

#A'''

'''B
#Generate a representation of the model architecture
modelArch = "5_layer_cnn.gv"
archTitle = "Five Layer CNN"
print(f"Saving model architecture to {modelArch}")
ann_viz(model, title=archTitle, filename=modelArch)
B'''

curRun = "run_1"

#'''C
#Generate graphs of accuracy vs time and loss vs time
#Then save the model
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

modelResults = 'models/' + curRun + '.h5'

model.save_weights(modelResults)
print(f"Saving model results to {modelResults}")
#C'''
