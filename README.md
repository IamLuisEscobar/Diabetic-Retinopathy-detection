# Diabetic-Retinopathy-detection
Project focus in the detection of diabetic retinopathy using some preprocessing techniques and Convolutional Neural Networks as classifier 
The objective of the project is to classify ocular fundus images as healthy or sick. Some preprocessing techniques were used to identify which of them provide better results. The classifier is a Convolutional Neural Network using as base model the InceptionV3 architecture. Transfer learning and data augmentation were used during the training process.

### About the code [listed in alphabetical order]

**Binary_patches.py**
-Use as input annotated images an with thar imformation the original image is cropped in the area of interest 

**Clipped_Final.py**
-Remove the innecesary black background from the images

**Color_Patches.py**
-Save patches of a dataset of images. The input are the patch size and the stride through the image
Evaluation_CNN.py
Image_duplicate.py
Json_Weights_Transfer.py
This file is the responsable of the training.

Keras_Fine_Tunning.py
Is a variant of the first file. Here we train the layers at the begginign and at the end
New_Clahe.py
RGB_mean.py
Resize_Average.py

