# Diabetic-Retinopathy-detection
### Abstract


Diabetic retinopathy is a disease that occurs as a result of damage to the blood vessels of the retina. This damage can cause partial or total blindness which is preventable with an early detection and treatment. The California Health Care Foundation CHCF recently found that among patients with diabetic retinopathy who were referred to a specialist by their primary care physicians only 23% of 288 patients with advanced retinal disease from four high-performing clinics ever made it to ophthalmology care. Patients fell out at every step of the process: 15% never learned about their disease, another 15% did not receive an appointment, 22% did not attend their appointments, and 25% opted out of treatment. With the projection of growth of the disease and the lack of specialists, it is necessary to develop a tool that helps in the detection of diabetic retinopathy. This research focus in the implementation of a Convolutional Neural Network, which uses retinal images as input and generates a prediction value for the disease at its output. This seeks to validate the reliability of Convolutional Neural Networks in the detection of diabetic retinopathy and thus propose its widespread use to reduce the workload of ophthalmologists by referring with them only to patients who really need it.
<p align="center">
  <img width="400" height="400" src="https://upload.wikimedia.org/wikipedia/commons/3/37/Fundus_photograph_of_normal_right_eye.jpg"
</p>

### Methodology
The following diagram shows the methodology used to classify ocular fundus images.
<p align="center">
  <img width="1000" height="150" src="https://github.com/IamLuisEscobar/Diabetic-Retinopathy-detection/blob/master/Picture1.png"
</p>

### Data

### Data preprocessing
**Step 1**
All the images were cropped to eliminate the black edges. Clipped_Final.py is the script resposable of the task 

**Step 2**
The images were normalized to reduce variation in light conditions.

**Step 3**
Contrast Limited Adaptative Histogram Equalization was used to imporove the contrast in the images. New_Clahe.py

### Training of ConvNet 
Four architectures were used to compare their results in the same dataset.

- **[1] Darshit Doshi et al**
- **[2] Kele Xu et al**
- **[3] ResNet 50**
- **[4] InceptionV3**

**Image augmentation**  
The following transformations were applied to each image to increase the dataset
rescale=1./255  
shear_range=0.2  
zoom_range=0.2  
rotation_range=20  
width_shift_range=0.2  
height_shift_range=0.2  
horizontal_flip=True

**Transfer learning**


**Tunning of Hyperparameters**  
Learning rate ...... 0.0001  
Batch size ......... 16  
Epochs ............. 30  
Activation functions ReLu and Sigmoid in the last layer

**Test of ConvNet with new data**


**Results and visualization**
https://github.com/keras-team/keras/issues/3477
This was a great resource when I got problems with the obtained predictions between predict and predict_generator. The answer provided by atortoricimontaperto resume the problem and provide solutions to avoid different results
**How loss is obtained in keras**
https://stackoverflow.com/questions/58159154/how-to-calculate-categorical-cross-entropy-by-hand

### About the code [listed in alphabetical order]

**Binary_patches.py**
-Use as input annotated images to crop from the original image an area of interest

**Clipped_Final.py**
-Remove the characteristic black background of the fundus images

**Color_Patches.py**
-Save patches of a set of images. Receive as input the patch size and the stride used through the image

**Evaluation_CNN.py**
-Code to evaluate the performance  of a trained model

**Image_duplicate.py**
-Search through a folder for possible duplicate images

**Json_Weights_Transfer.py**
-Train a base network, default is InceptionV3, applying transfer learning and data augmentation

**Keras_Fine_Tunning.py**
-Train a base network fine tuning the layers

**New_Clahe.py**
Apply Clahe filter to a set of images

**RGB_mean.py**
-Zero center and normalize the images used for train, validation and test

**Resize_Average.py**
-Resize a set of images obtaining the average though all the images in a given folder 
Citation
[1] D. Doshi, A. Shenoy, D. Sidhpura and P. Gharpure, "Diabetic Retinopathy detection using deep convolutional neural networks," International Conference on Computing, Analytics and Security Trends , 19-21 Dec 2016.
[2] K. F. D. &. M. H. Xu, "Deep convolutional neural network-based early automated detection of diabetic retinopathy using fundus image," Molecules, pp. 22(12), 2054., 2017. 
