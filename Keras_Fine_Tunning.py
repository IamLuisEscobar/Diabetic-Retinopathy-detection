from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.utils.training_utils import multi_gpu_model
from keras.optimizers import SGD
import tensorflow as tf

# dimensions of images.
img_width, img_height = 150, 150

train_data_dir='/home/luis/DataSets/IMO+Messidor+IDRID/Test_PreProcessing/Kaggle_Clahe/training'
validation_data_dir='/home/luis/DataSets/IMO+Messidor+IDRID/Test_PreProcessing/Kaggle_Clahe/testing'
testing_data_dir='/home/luis/DataSets/IMO+Messidor+IDRID/Test_PreProcessing/Kaggle_Clahe/validation'

#Defining some parameters
nb_train_samples = 1182
nb_validation_samples = 329
nb_epoch = 5
batch=8
G=2

# create the base model with imagenet weights
base_model = InceptionV3(weights='imagenet', include_top=False)

# Adding a layer to retrain the model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
# Prediction for two classes
predictions = Dense(2, activation='softmax')(x)

model = Model(input=base_model.input, output=predictions)

# First are train only the top layers

for layer in base_model.layers:
    layer.trainable = False

# Multiple GPU computing 
if G <= 1:
        print("[INFO] training with 1 GPU...")
        train_model = model
# otherwise, we are compiling using multiple GPUs
else:
        print("[INFO] training with {} GPUs...".format(G))
 
        # we'll store a copy of the model on *every* GPU and then combine
        # the results from the gradient updates on the CPU
        with tf.device("/cpu:0"):
                # initialize the model
                train_model = model
        
        # make the model parallel
        train_model = multi_gpu_model(model,gpus=G)

# Compile the model 
train_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation
train_datagen=ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch,
    class_mode='categorical'
)


history = train_model.fit_generator(
    train_generator,
    nb_epoch=nb_epoch,
    samples_per_epoch=nb_train_samples//batch,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples//batch) 

# The top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. 

# we chose to train the top 2 inception blocks
for layer in train_model.layers[:172]:
   layer.trainable = False
for layer in train_model.layers[172:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use a new optimizer with low learning rate
train_model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks)

train_model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples//batch,
        nb_epoch=50,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples//batch)
#Finally we save our model as json file and the weights
model_json=model.to_json()
with open('Inception_model_01.json','w') as json_file:
    json_file.write(model_json)
model.save_weights('Inception_weights_01.h5')
print ('Saved model and weights to disk')


