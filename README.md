# Diabetic-Retinopathy-detection
Project focus in the detection of diabetic retinopathy using some preprocessing techniques and Convolutional Neural Networks as classifier 
The objective of the project is to classify ocular fundus images as healthy or sick. Some preprocessing techniques were used to identify which of them provide better results. The classifier is a Convolutional Neural Network using as base model the InceptionV3 architecture. Transfer learning and data augmentation were used during the training process.

About the code

Json_Weights_Transfer.py
This file is the responsable of the training.

Keras_Fine_Tunning.py
Is a variant of the first file. Here we train the layers at the begginign and at the end

Evaluation_CNN.py
Code to evaluate images with the model trained
