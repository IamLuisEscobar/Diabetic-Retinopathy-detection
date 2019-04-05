from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.utils.training_utils import multi_gpu_model
import tensorflow as tf

# dimensions of our images.
img_width, img_height = 299, 299

#train_data_dir = '/Users/michael/testdata/train' #contains two classes cats and dogs
#validation_data_dir = '/Users/michael/testdata/validation' #contains two classes cats and dogs

train_data_dir='/home/luis/DataSets/IMO+Messidor+IDRID/Test_PreProcessing/Kaggle_Clahe/training'
validation_data_dir='/home/luis/DataSets/IMO+Messidor+IDRID/Test_PreProcessing/Kaggle_Clahe/testing'
testing_data_dir='/home/luis/DataSets/IMO+Messidor+IDRID/Test_PreProcessing/Kaggle_Clahe/validation'

nb_train_samples = 1182
nb_validation_samples = 329

#nb_train_samples = 1200
#nb_validation_samples = 800
nb_epoch = 5
batch=8
G=2
# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(2, activation='softmax')(x)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# check to see if we are compiling using just a single GPU
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

# compile the model (should be done *after* setting layers to non-trainable)
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
train_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
'''
train_datagen=ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

val_datagen=ImageDataGenerator(
        rescale=1./255)
train_generator=train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch,
        class_mode='binary')
validation_generator=val_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height,img_width),
        batch_size=batch,
        class_mode='binary')
testing_generator=val_datagen.flow_from_directory(
        testing_data_dir,
        target_size=(img_height,img_width),
        batch_size=batch,
        class_mode='binary')


'''
# prepare data augmentation configuration
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

print "start history model"
history = train_model.fit_generator(
    train_generator,
    nb_epoch=nb_epoch,
    samples_per_epoch=nb_train_samples//batch,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples//batch) #1020

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
for layer in train_model.layers[:172]:
   layer.trainable = False
for layer in train_model.layers[172:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
train_model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
#model.fit_generator(...)
# fine-tune the model
train_model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples//batch,
        nb_epoch=50,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples//batch)

model_json=model.to_json()
with open('Inception_model_01.json','w') as json_file:
    json_file.write(model_json)
model.save_weights('Inception_weights_01.h5')
print ('Saved model and weights to disk')


