#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 11:05:12 2019

@author: luishdz
"""
#Window slider for patches
import cv2
import os

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
    
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
def name_patch(ii,window,filename,save_path):
    if ii<10:
        save_name=os.path.join(save_path,'Patch_' + filename + '_00' + str(ii) + '.jpg')
        cv2.imwrite(save_name,window)
    elif 10 <= ii <= 99:
        save_name=os.path.join(save_path,'Patch_' + filename + '_0' + str(ii) + '.jpg')
        cv2.imwrite(save_name,window)
    elif ii > 99:
        save_name=os.path.join(save_path,'Patch_' + filename + '_' + str(ii) + '.jpg')
        cv2.imwrite(save_name,window)
        
def save_patch_coordinate(image,stepSize,winW,winH,filename,save_path):
    save_count=0
    for (x, y, window) in sliding_window(image, stepSize, windowSize=(winW, winH)):
            mapping=image[x:x+winW,y:y+winH]
            count=0
            if mapping.shape[0]==150 and mapping.shape[1]==150:
                if 0 in mapping:
                    for i in range(mapping.shape[0]):
                        for j in range(mapping.shape[1]):
                            if mapping[i,j][0] == 0:
                                count=count+1
                    if count < 16875:
                        name_patch(save_count,mapping,filename,save_path)
                        save_count+=1
                else:
                    name_patch(save_count,mapping,filename,save_path)
                    save_count += 1
            if window.shape[0] != winH or window.shape[1] != winW:
                    continue 


#%%
path_color='/home/luis/DataSets/Annotations/EX_healthy_Clipped/'
save_path='/home/luis/DataSets/Patches/e_optha/healthy'
files_color=getListOfFiles(path_color)

(winW, winH) = (150, 150)
stepSize=32
patch=[]
#%%
for fname in files_color:
    print('Reading a new pair of images')
    color_image=cv2.imread(fname)
    filename=os.path.split(fname)[1]
    filename=os.path.splitext(filename)[0]
    print ('Sliding window')
    save_patch_coordinate(color_image,stepSize,winW,winH,filename,save_path)
    
                                                   
    
