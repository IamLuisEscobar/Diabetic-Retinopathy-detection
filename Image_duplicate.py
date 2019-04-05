#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 19:14:57 2019

@author: luishdz
"""

#import numpy as np
import cv2
import os

def read_images_from_folder(folder):
    names=[]
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            names.append(filename)
    return names

folder='/home/luis/DataSets/Test/'
names=read_images_from_folder(folder)
count=len(names)
for fnames in names:
    img1=cv2.imread(folder+fnames,0)
    for fnames2 in names:
        if fnames==fnames2:
            pass
        else:
            img2=cv2.imread(folder+fnames2,0)
            if img1.shape==img2.shape:
                difference = img1-img2
                if cv2.countNonZero(difference) == 0:
                    print (folder+fnames)
                    print (folder+fnames2)
                else:
                    pass
    print count
    count-=1   
#%%
