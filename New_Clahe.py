#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 12:21:59 2019

@author: luishdz
"""
import cv2
import os

def read_images_from_folder(folder):
    names=[]
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            names.append(filename)
    return names

def clahe(image_path,gridsize):
    bgr = cv2.imread(image_path)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(gridsize,gridsize))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr

folder=[]
training_class_01 = '/home/luis/DataSets/Patches/Test03/train/MA'
training_class_02 = '/home/luis/DataSets/Patches/Test03/train/healthy'
validation_class_01 = '/home/luis/DataSets/Patches/Test03/val/MA'
validation_class_02 = '/home/luis/DataSets/Patches/Test03/val/healthy'
testing_class_01 = '/home/luis/DataSets/Patches/Test03/test/MA'
testing_class_02 = '/home/luis/DataSets/Patches/Test03/test/healthy'
folder.append(training_class_01)
folder.append(training_class_02)
folder.append(validation_class_01)
folder.append(validation_class_02)
folder.append(testing_class_01)
folder.append(testing_class_02)

'''
for path_names in folder:
    names=read_images_from_folder(path_names)
    for fnames in names:
        bgr=clahe(os.path.join(path_names,fnames),8)
        final='Clahe_'+fnames
        cv2.imwrite(os.path.join(path_names,final),bgr)
    print path_names
'''

for path_names in folder:
    names=read_images_from_folder(path_names)
    for fnames in names:
        try:
            bgr=clahe(os.path.join(path_names,fnames),8)
            final='Clahe_'+fnames
            cv2.imwrite(os.path.join(path_names,final),bgr)
        except:
            print ('Fail')
            print os.path.join(path_names,fnames)
            break
    print ('Successful')
    print path_names
