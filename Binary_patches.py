#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:53:51 2019

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

def patch_coordinate(image,stepSize,winW,winH):
    patch=[]
    for (x, y, window) in sliding_window(image, stepSize, windowSize=(winW, winH)):
            mapping=black_image[x:x+winW,y:y+winH]
            count=0
            if 255 in mapping:
                for i in range(mapping.shape[0]):
                    for j in range(mapping.shape[1]):
                        if mapping[i,j] == 255:
                            count=count+1
                if count > 10:
                    patch.append((x,x+winW,y,y+winH))
            if window.shape[0] != winH or window.shape[1] != winW:
                continue 
    return patch

def save_patch(patch,image,filename,save_path):
    for ii in range(len(patch)):
        x=patch[ii][0]
        winW=patch[ii][1]
        y=patch[ii][2]
        winH=patch[ii][3]
        window=color_image[x:winW,y:winH]
        if ii<10:
            save_name=os.path.join(save_path,'Patch_' + filename + '_00' + str(ii) + '.jpg')
            cv2.imwrite(save_name,window)
        elif 10 <= ii <= 99:
            save_name=os.path.join(save_path,'Patch_' + filename + '_0' + str(ii) + '.jpg')
            cv2.imwrite(save_name,window)
        elif ii > 99:
            save_name=os.path.join(save_path,'Patch_' + filename + '_' + str(ii) + '.jpg')
            cv2.imwrite(save_name,window)

#%%

path_black='/home/luis/DataSets/Annotations/e_optha_MA/Annotation_MA'
path_color='/home/luis/DataSets/Annotations/e_optha_MA/MA'
save_path='/home/luis/DataSets/Patches/e_optha/MA'

#%%
#path_black='/Users/luishdz/Downloads/e_optha_EX/Annotation_EX/'
#path_color='/Users/luishdz/Downloads/e_optha_EX/EX/'
#save_path='/Users/luishdz/Downloads'
files_black=getListOfFiles(path_black)
files_color=getListOfFiles(path_color)

(winW, winH) = (150, 150)
stepSize=32
patch=[]
#%%
for fname in files_black:
    print('Reading a new pair of images')
    black_image=cv2.imread(fname,0)
    folder = os.path.split(fname)[0]
    folder = os.path.split(folder)[1]
    filename = os.path.split(fname)[1]
    filename = os.path.splitext(filename)[0]
    filename = filename.replace('_EX','')
    name1=os.path.join(path_color,folder,filename+'.jpg')
    name2=os.path.join(path_color,folder,filename+'.JPG')
    if os.path.isfile(name1):
        color_image=cv2.imread(name1)
    elif os.path.isfile(name2):
        color_image=cv2.imread(name2)
    else:
        print ('Here be dragons')
    #First verify that both images have the same size
    if color_image.shape[0]==black_image.shape[0] and color_image.shape[1]==black_image.shape[1]:
        print ('Sliding window')
        patch=patch_coordinate(black_image,stepSize,winW,winH)
        save_patch(patch,color_image,filename,save_path)                                                
    else:
        print ('Shape is not the same')
        print fname
        print filename
#%%        
    
		









 
