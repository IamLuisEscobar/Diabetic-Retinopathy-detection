#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 11:45:05 2019

@author: luishdz
"""
from __future__ import division
import numpy as np
import os
import cv2

def read_images_from_folder(folder):
    names=[]
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.tif'):
            names.append(filename)
    return names

#Función para obtener media y desviación estándar de un data set dado. Se debe
#utilizar con el dataset de entrenamiento y esos mismo valores se utilizarán
#con el data set del validación y testing. NO se deben obtener nuevos valores
#para ellos    
def mean_std(img):
    b,g,r=cv2.split(img)
    #b = np.asarray(b).astype(float), b=b/255
    #g = np.asarray(g).astype(float), g=g/255
    #r = np.asarray(r).astype(float), r=r/255
    b = b / 255
    g = g / 255
    r = r / 255
    b_mean = np.mean(b)
    b=b-b_mean
    b_std = np.std(b)
    g_mean = np.mean(g) 
    g=g-g_mean
    g_std = np.std(g)
    r_mean = np.mean(r)
    r=r-r_mean
    r_std = np.std(r)
    b=(b_mean,b_std)
    g=(g_mean,g_std)
    r=(r_mean,r_std)
    return b, g, r

#Se centra la media de los datos a cero 
def zero_center(img, B, G, R):
    b,g,r=cv2.split(img)
    b = b / 255
    g = g / 255
    r = r / 255    
    b=b-B[0]
    g=g-G[0]
    r=r-R[0]

    b=np.clip(b, -1.0, 1.0)
    g=np.clip(g, -1.0, 1.0)
    r=np.clip(r, -1.0, 1.0)
    
    b=(b + 1.0) / 2.0
    g=(g + 1.0) / 2.0
    r=(r + 1.0) / 2.0
    
    b=(b * 255 + 0.5).astype(np.uint8)
    g=(g * 255 + 0.5).astype(np.uint8)
    r=(r * 255 + 0.5).astype(np.uint8)
    
    img=cv2.merge((b,g,r))    
    return img

#Se centran los datos a cero y se divide entre la desviación estándar promedio. Importante recordar que 
#el calculo de la media y la desviación estándar sólo se realiza con el set de entrenamiento. Esos mismos datos son utilizados
#con el set de validación y testing 
def norm_image(img, B, G, R):
    b,g,r=cv2.split(img)
    b = b / 255
    g = g / 255
    r = r / 255    
    b=b-B[0]
    b=b/np.std(b) 
    g=g-G[0]
    g=g/np.std(g) 
    r=r-R[0]
    r=r/np.std(r) 

    b=np.clip(b, -1.0, 1.0)
    g=np.clip(g, -1.0, 1.0)
    r=np.clip(r, -1.0, 1.0)
    
    b=(b + 1.0) / 2.0
    g=(g + 1.0) / 2.0
    r=(r + 1.0) / 2.0
    
    b=(b * 255 + 0.5).astype(np.uint8)
    g=(g * 255 + 0.5).astype(np.uint8)
    r=(r * 255 + 0.5).astype(np.uint8)
    
    img=cv2.merge((b,g,r))
    return img

training_class_01 = '/home/luis/DataSets/Patches/Test02/train/healthy/'
training_class_02 = '/home/luis/DataSets/Patches/Test02/train/MA/'
names_training_class_01=read_images_from_folder(training_class_01)
names_training_class_02=read_images_from_folder(training_class_02)


validation_class_01 = '/home/luis/DataSets/Patches/Test02/val/healthy/'
validation_class_02 = '/home/luis/DataSets/Patches/Test02/val/MA/'
names_validation_class_01=read_images_from_folder(validation_class_01)
names_validation_class_02=read_images_from_folder(validation_class_02)


#Emplear si se cuenta con un folder de testing
testing_class_01 = '/home/luis/DataSets/Patches/Test02/test/healthy/'
testing_class_02 = '/home/luis/DataSets/Patches/Test02/test/MA/'
names_testing_class_01=read_images_from_folder(testing_class_01)
names_testing_class_02=read_images_from_folder(testing_class_02)

#Se crean listas para obtener el promedio y desviación estándar del set de entrenamiento
B_mean=[]
B_std=[]
G_mean=[]
G_std=[]
R_mean=[]
R_std=[]

#%%
#Iteramos sobre la primer clase [Normal-Mild] para obtener media y desviación estándar de cada imagen
count=len(names_training_class_01)

for fnames in names_training_class_01:     
    try:
        img = cv2.imread(training_class_01+fnames)
        b,g,r= mean_std(img)
        B_mean.append(b[0]), B_std.append(b[1])
        G_mean.append(g[0]), G_std.append(g[1])
        R_mean.append(r[0]), R_std.append(r[1])
        print count
        count -= 1
    except:
        print(training_class_01+fnames)
#%%  
#Iteramos sobre la segunda clase [Moderate-Severe] para obtener media y desviación estándar de cada imagen        
count=(len(names_training_class_02))
for fnames in names_training_class_02:
    try:
        img = cv2.imread(training_class_02+fnames)
        b,g,r= mean_std(img)
        B_mean.append(b[0]), B_std.append(b[1])
        G_mean.append(g[0]), G_std.append(g[1])
        R_mean.append(r[0]), R_std.append(r[1])
        print count
        count -= 1
    except:
        print(training_class_02+fnames)
            
#%%
#A partir de las listas generadas se obtienen un promedio y una desviación estándar [ambos promedio de todas las imágenes]
B_mean=sum(B_mean)/len(B_mean)
B_std=sum(B_std)/len(B_std)
G_mean=sum(G_mean)/len(G_mean) 
G_std=sum(G_std)/len(G_std)
R_mean=sum(R_mean)/len(R_mean) 
R_std=sum(R_std)/len(R_std)

B=(B_mean,B_std)
G=(G_mean,G_std)
R=(R_mean,R_std)
#%%
count=(len(names_training_class_01))
#Iteración en primer clase de entrenamiento para centrar datos
for fnames in names_training_class_01:
    try:
        img = cv2.imread(training_class_01+fnames)
        img=zero_center(img,B,G,R)
        cv2.imwrite(training_class_01 + 'Zero_'+fnames,img)
        print count
        count -= 1 
    except:
        print(training_class_01+fnames)
        #print('New')
        #print fnames
        break
#%%
count=(len(names_training_class_02))
#Iteración en segunda clase de entrenemiento para centrar datos        
for fnames in names_training_class_02:
    try:
        img = cv2.imread(training_class_02+fnames)
        img=zero_center(img,B,G,R)
        cv2.imwrite(training_class_02 + 'Zero_'+fnames,img)
        print count
        count -= 1
    except:
        print(training_class_02+fnames)
        break

count=len(names_validation_class_01)
#Iteración en primera clase de validación para centrar datos              
for fnames in names_validation_class_01:
    try:
        img = cv2.imread(validation_class_01+fnames)
        img=zero_center(img,B,G,R)
        cv2.imwrite(validation_class_01 + 'Zero_'+fnames,img)
        print count
        count -= 1
    except:
        print(validation_class_01+fnames)

#Iteración en segunda clase de validación para centrar datos              
count=len(names_validation_class_02)
for fnames in names_validation_class_02:
    try:
        img = cv2.imread(validation_class_02+fnames)
        img=zero_center(img,B,G,R)
        cv2.imwrite(validation_class_02 + 'Zero_'+fnames,img)
        print count
        count -= 1
    except:
        print(validation_class_01+fnames)        
       

#Si existe folder de testing utilizar las últimas líneas      
       
#Iteración en primera clase de testing para centrar datos              
count=len(names_testing_class_01)
for fnames in names_testing_class_01:
    try:
        img = cv2.imread(testing_class_01+fnames)
        img=zero_center(img,B,G,R)
        cv2.imwrite(testing_class_01 + 'Zero_'+fnames,img)
        print count
        count -= 1
    except:
        print(testing_class_01+fnames)

#Iteración en segunda clase de testing para centrar datos              
count=len(names_testing_class_02)
for fnames in names_testing_class_02:
    try:
        img = cv2.imread(testing_class_02+fnames)
        img=zero_center(img,B,G,R)
        cv2.imwrite(testing_class_02 + 'Zero_'+fnames,img)
        print count
        count -= 1
    except:
        print(testing_class_02+fnames)        
                
