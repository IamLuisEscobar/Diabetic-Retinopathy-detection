# Diabetic-Retinopathy-detection
Project focus in the detection of diabetic retinopathy using some preprocessing techniques and Convolutional Neural Network as classifier. 
The objective of the project is to classify ocular fundus images as healthy or sick. Some preprocessing techniques were used to identify which of them provide better results. The classifier is a Convolutional Neural Network using as base model the InceptionV3 architecture. Transfer learning and data augmentation were used during the training process.

### About the code [listed in alphabetical order]

**Binary_patches.py**
-Use as input annotated images cropping from the original image an area of interest

**Clipped_Final.py**
-Remove the black background from the images

**Color_Patches.py**
-Save patches of a set of images. Receive as input the patch size and the stride through the image

**Evaluation_CNN.py**
-Code to evaluate the performance  of a trained model

**Image_duplicate.py**
-Search through a folder for possible duplicate images

**Json_Weights_Transfer.py**
-Train a base network, default is InceptionV3, applying transfer learning and data augmentation

**Keras_Fine_Tunning.py**
-Train a base network fine tuning the layers

**New_Clahe.py**
Apply Clahe to a set of images

**RGB_mean.py**
-Zero center and normalize the data [in this case images] used for train, validate and test

**Resize_Average.py**
Resize a set of images obtaining the average though all the images in a given folder 
