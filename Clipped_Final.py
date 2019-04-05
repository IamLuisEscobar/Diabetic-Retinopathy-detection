#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 20:08:17 2019

@author: luishdz
"""

import os
import cv2

def read_images_from_folder(folder):
    names=[]
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.tif'):
            names.append(filename)
    return names

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


print ('Saving folders data')
folder = '/home/luis/DataSets/Annotations/e_optha_EX/healthy/'
names = getListOfFiles(folder)
S_folder=('/home/luis/DataSets/Annotations/EX_healthy_Clipped/')
#Load image
count=len(names)
print ('Loading images')
for filename in names:
    img=cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Find center of image
    rows = img.shape[0]-1;
    center_row=int(rows/2)
    columns = img.shape[1]-1;
    center_column=int(columns/2)
    
    for i in range (center_row):
        if gray[i,center_column] > 10:
            if i > 0:
                north=i
                break
            else:
                north=None
                break
        else:
            pass    
    
    for i in range (rows,center_row,-1):
        if gray[i,center_column] > 10:
            if i < rows:
                south=i
                break
            else:
                south=None
                break
        else:
            pass
    
    for i in range (center_column):
        if gray[center_row,i] > 10:
            if i > 0:
                west=i
                break
            else:
                west=None
                break
        else:
            pass
    
    for i in range (columns,center_column,-1):
        if gray[center_row,i] > 10:
            if i < columns:
                east=i
                break
            else:
                east=None
                break
        else:
            pass
    
    if south == None:
        south=rows
    else:
        pass
    if north == None:
        north=0
    else:
        pass
    
    if east == None:
        east=columns
    else:
        pass
        
    if west == None:
        west=0
    else:
        pass
    
    
    img2=img[north:south,west:east]
    S_filename_01=os.path.split(filename)[1]
    S_filename='Clipped_' + os.path.splitext(S_filename_01)[0]+'.jpg'
    cv2.imwrite(os.path.join(S_folder,S_filename),img2)
    print count
    count-=1

#%%    
