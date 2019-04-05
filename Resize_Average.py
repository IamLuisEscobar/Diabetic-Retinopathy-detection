#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 18:51:43 2019

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

folder='/home/luis/DataSets/IDRID+Messidor/Original/Moderate_Severe/'
save_folder='/home/luis/DataSets/IDRID+Messidor/Original/Moderate_Severe/'
names=read_images_from_folder(folder)
#Load image
list_columns=[]
list_rows=[]
count=len(names)
#%%
for filenames in names:
    img=cv2.imread(folder+filenames)
    rows=img.shape[0]
    columns=img.shape[1]
    list_columns.append(columns)
    list_rows.append(rows)
    Average_columns=int(sum(list_columns)/len(list_columns))
    Average_rows=int(sum(list_rows)/len(list_rows))
    print count
    count-=1
#%%
count=len(names)
for filenames in names:
    img=cv2.imread(folder+filenames)
    img2=cv2.resize(img,(Average_columns,Average_rows), interpolation=cv2.INTER_CUBIC)
    S_filename=save_folder + 'Resize_' + filenames 
    cv2.imwrite(S_filename,img2)
    print count
    count-=1
print Average_columns,Average_rows
#%%
