#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 01:25:46 2019

@author: luishdz
"""
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

img_width,img_height=150,150

train_data_dir='/home/luis/DataSets/Patches/Test03/train'
validation_data_dir='/home/luis/DataSets/Patches/Test03/test'
testing_data_dir='/home/luis/DataSets/Patches/Test03/val'

train_samples = 45831
validation_samples = 5739
test_samples = 5720
G=2
epochs=8
batch_size=16
base_model=applications.InceptionV3(weights='imagenet',include_top=False,
                                    input_shape=(img_width,img_height,3))

model_top=Sequential()
model_top.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:],
                                     data_format=None))
model_top.add(Dense(256,activation='relu'))
model_top.add(Dropout(0.5))
#model_top.add(Dense(1,activation='sigmoid'))
model_top.add(Dense(2,activation='sigmoid'))
model=Model(inputs=base_model.input,outputs=model_top(base_model.output))

# check to see if we are compiling using just a single GPU
if G <= 1:
        print("[INFO] training with 1 GPU...")
        train_model = model
# otherwise, we are compiling using multiple GPUs
else:
        print("[INFO] training with {} GPUs...".format(G))
 
        # we'll store a copy of the model on *every* GPU and then combine
        # the results from the gradient updates on the CPU
        with tf.device("/cpu:0"):
                # initialize the model
                train_model = model
        
        # make the model parallel
        train_model = multi_gpu_model(model,gpus=G)

train_model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9,beta_2=0.999,epsilon=1e-08,
                             decay=0.0),loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen=ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

val_datagen=ImageDataGenerator(
        rescale=1./255)
train_generator=train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')
validation_generator=val_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height,img_width),
        batch_size=batch_size,
        class_mode='categorical')
testing_generator=val_datagen.flow_from_directory(
        testing_data_dir,
        target_size=(img_height,img_width),
        batch_size=batch_size,
        class_mode='categorical')

'''
# checkpoint save if improve
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
'''

# checkpoint save the best
filepath="weights_best.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


H=train_model.fit_generator(
        train_generator,
        steps_per_epoch=train_samples//batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_samples//batch_size) 
                                    
#Es necesario nombrar de diferente modo al modelo del gpu
model_json=model.to_json()
with open('InceptionV3_model_01.json','w') as json_file:
    json_file.write(model_json)
model.save_weights('InceptionV3_weights_01.h5')
print ('Saved model and weights to disk')

# plot the training loss and accuracy
N = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('Plot_test_01.png')

