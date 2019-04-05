#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 17:11:38 2019

@author: luishdz
"""

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import model_from_json
import numpy as np
import cv2
import os

#Function to load all the images in a given folder including possible subfolders
def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    Filter_list=[]
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    for entry in allFiles:
        if entry.endswith('.png') or entry.endswith('.jpg') or entry.endswith('.JPG'):
            Filter_list.append(entry)
        else:
            pass
    return Filter_list

#Getting paths
Filter_list=getListOfFiles('/home/luis/DataSets/Patches/Test03/val/MA')
path_to_model = '/home/luis/DataSets/Patches/Test03/InceptionV3_model_01.json'
path_to_weights ='/home/luis/DataSets/Patches/Test03/InceptionV3_weights_01.h5'


 
print("[INFO] loading network...")
# load json and create model
json_file = open(path_to_model, 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
# load weights into new model
model.load_weights(path_to_weights)
count=0
count_healthy=0
count_notHealthy=0
# For loop to classify a set of images
for fnames in Filter_list:
    image = cv2.imread(fnames)
    if image.shape[0] != 150 or image.shape[1] != 150:
        continue
    orig = image.copy()
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    (notHealthy, healthy) = model.predict(image)[0]
    label = "Healthy" if healthy > notHealthy else "Not Healthy"
    proba = healthy if healthy > notHealthy else notHealthy
    label = "{}: {:.2f}%".format(label, proba * 100)
    if healthy > notHealthy:
        count_healthy+=1
    else:
        count_notHealthy+=1
    count +=1
    if count ==100:
        break
#First sensitivity and specificity test
print ('Healthy')
print count_healthy
print ('Not Healthy')
print count_notHealthy


 
