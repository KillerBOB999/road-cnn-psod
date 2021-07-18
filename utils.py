import numpy as np
import pandas as pd

import sklearn
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.random import set_seed

from typing import Tuple, List
from tqdm import tqdm
import os

#Seeding random state to 13 for reproducibility
seed = 13
np.random.seed(seed)
set_seed(seed)

#Declare Callbacks: stop training if accuracy doesn't rise 1% within 3 epochs
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor = "val_loss",
        patience = 3,
        verbose = 1,
        restore_best_weights = True,
        min_delta = 0.01
    )
]

#Helper Function: Return the paths to all jpg files found within a directory
def getImageDirs(root: str = "data"):
    imageDirs = []
    for subDirectory, directory, files in os.walk(root):
        for file in files:
            if file[-4:] == ".jpg":
                path = os.path.join(subDirectory, file)         
                imageDirs.append(path)
    return imageDirs

#Helper Function: Return the class weights given a list of classes
def getClassWeightsFromLabels(labels: List[int]):
    weights = sklearn.utils.class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
    return {0: weights[0], 1: weights[1]}

#Helper Function: Return the img paths and classes in seperate lists given a txt file from the LABELS folder
def getDirsAndClasses(root: str, file: str) -> Tuple[List[str], List[int]]:
    imageDirs = []
    classes = []
    line = ""
    with open(root + file, "r") as f:
        for line in tqdm(f):
            imageDir, clazz = line.split()
            imageDirs.append(imageDir)
            classes.append(int(clazz))
    return imageDirs, classes

def makeDataset(pathFromCwdToDataRoot = "Data") -> pd.DataFrame:
    #Get the Train Dataset using split from the LABELS folder
    root = f'{os.getcwd()}/{pathFromCwdToDataRoot}/CNR-EXT-150x150'
    imageDirs, classes = getDirsAndClasses(root, "/LABELS/train.txt")
    patchesRoot = root + "/PATCHES/"
    train = pd.DataFrame([
                {
                    "image": patchesRoot + filename,
                    "class": "free" if clazz == 0 else "busy"
                }
                for filename, clazz in tqdm(zip(imageDirs, classes))
        ])
    #Now Get Test
    imageDirs, classes = getDirsAndClasses(root, "/LABELS/test.txt")
    test = pd.DataFrame([
                {
                    "image": patchesRoot + filename,
                    "class": "free" if clazz == 0 else "busy"
                }
                for filename, clazz in tqdm(zip(imageDirs, classes))
        ])
    
    # Combine them together to get the full dataset
    dataset = train.append(test)
    
    return dataset

# Helper Function: Create the k-fold splits and calculate the class weights
def makeFolds(dataset: pd.DataFrame, n_folds: int = 1, batch_size: int = 128, target_size: Tuple[int] = (150, 150)):
    # Choose number of folds (1=normal experiment)
    # Make the k folds
    kFCV_sets=[]

    busy_samples=dataset.loc[dataset["class"] == "busy"]
    free_samples=dataset.loc[dataset["class"] == "free"]

    # If we don't want to perform k-folding
    if n_folds == 1:
        busy = [i for i in range(int(len(busy_train)*0.9))],[i for i in range(int(len(busy_train)*0.9), len(busy_train))]
        free = [i for i in range(int(len(free_train)*0.9))],[i for i in range(int(len(free_train)*0.9), len(free_train))]
    # We do want to perform k-folding
    else:
        busy_kf = sklearn.model_selection.KFold(n_splits = n_folds)
        free_kf = sklearn.model_selection.KFold(n_splits = n_folds)

    # Iterate through the folds and make the data generators for the models to use
    for k in range(n_folds):
        if n_folds != 1:
            busy = next(busy_kf.split(busy_samples), None)
            free = next(busy_kf.split(free_samples), None)
        busy_train, free_train = busy_samples.iloc[busy[0]], free_samples.iloc[free[0]]
        busy_train, busy_val = busy_train[:int(len(busy_train)*0.9)], busy_train[int(len(busy_train)*0.9):]
        free_train, free_val = free_train[:int(len(free_train)*0.9)], free_train[int(len(free_train)*0.9):]

        train = busy_train.append(free_train)
        val = busy_val.append(free_val)
        test = busy_samples.iloc[busy[1]].append(free_samples.iloc[free[1]])

        #Declare data generators and preprocessing
        train_datagen = ImageDataGenerator(
            #Augment data with random flips, normalize each sample's input
            vertical_flip = True,
            horizontal_flip = True,
            rescale = 1.0 / 255.0,
            samplewise_std_normalization = True
        )
        train_generator = train_datagen.flow_from_dataframe(
            directory = None, #none since the df has absolute paths
            dataframe = train,
            x_col = "image",
            y_col = "class",
            validate_filenames = False, #faster for huge datasets
            target_size = target_size,
            color_mode = "rgb",
            batch_size = batch_size,
            class_mode = "binary",
            shuffle = True
        )

        test_datagen = ImageDataGenerator(
            samplewise_std_normalization = True
        )
        test_generator = test_datagen.flow_from_dataframe(
            directory = None,
            dataframe = test,
            x_col = "image",
            y_col = "class",
            validate_filenames = False,
            target_size = target_size,
            color_mode = "rgb",
            batch_size = batch_size,
            class_mode = "binary",
            shuffle = True
        )
        val_generator = test_datagen.flow_from_dataframe(
            directory = None,
            dataframe = val,
            x_col = "image",
            y_col = "class",
            validate_filenames = False,
            target_size = target_size,
            color_mode = "rgb",
            batch_size = batch_size,
            class_mode = "binary",
            shuffle = True
        )
        
        print()

        kFCV_sets.append([train_generator, test_generator, val_generator])
        
    #Extract Class Weights (Weights will be the same for all folds)
    classes = list(train["class"])
    weights_dict = getClassWeightsFromLabels(classes)
    
    return kFCV_sets, weights_dict

#Helper Function: Create a TF/Keras model
def makeModel(inputShape: Tuple[int], modelName:str = '') -> keras.Model:
    model = None
    
    if modelName == "AlexNet":
        model = makeAlexNet(inputShape=inputShape, modelName=modelName)
    elif modelName in ["InceptionResNetV2","MobileNetV2","ResNet50V2","DenseNet121","DenseNet201","NASNetLarge"]: 
        model = makePrebuiltModel(inputShape=inputShape, modelName=modelName)
    elif modelName == "SimpleNet":
        model = makeSimpleNet(inputShape=inputShape, modelName=modelName)
    elif modelName == "SimpleDenseNet":
        model = makeSimpleDenseNet(inputShape=inputShape, modelName=modelName)
    elif modelName == "SimpleResNet":
        model = makeSimpleResNet(inputShape=inputShape, modelName=modelName)
    elif modelName == "mAlexNet":
        model = makemAlexNet(inputShape=inputShape, modelName=modelName)
    elif modelName == "mDenseNet":
        model = makemDenseNet(inputShape=inputShape, modelName=modelName)
    else:
        raise("Unknown model name given. Available names are AlexNet, InceptionResNetV2, MobileNetV2, ResNet50V2, DenseNet121, DenseNet201, NASNetLarge, SimpleNet, SimpleDenseNet, SimpleResNet, mAlexNet, and mDenseNet.")
    
    return model

##################################################################################################

"""
All of the following functions define the various models that can be used.
Each has the same signature:

------------------------ Params ------------------------
    inputShape: Tuple[int]      The shape of the images
    modelName: str              The name of the model
--------------------------------------------------------
"""

#Helper Function: Create an AlexNet model
def makeAlexNet(inputShape: Tuple[int], modelName: str ='') -> keras.Model:
    #Build Model
    AlexNet = Sequential(name=modelName)

    AlexNet.add(tf.keras.layers.experimental.preprocessing.Resizing(height=224, width=224))
    
    #1st Convolutional Layer
    AlexNet.add(Convolution2D(filters=96, input_shape=inputShape, kernel_size=(11,11), strides=(4,4), padding='same'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    #2nd Convolutional Layer
    AlexNet.add(Convolution2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    #3rd Convolutional Layer
    AlexNet.add(Convolution2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))

    #4th Convolutional Layer
    AlexNet.add(Convolution2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))

    #5th Convolutional Layer
    AlexNet.add(Convolution2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    #Passing it to a Fully Connected layer
    AlexNet.add(Flatten())
    # 1st Fully Connected Layer
    AlexNet.add(Dense(4096, input_shape=(32,32,3,)))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    AlexNet.add(Dropout(0.4))

    #2nd Fully Connected Layer
    AlexNet.add(Dense(4096))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    #Add Dropout
    AlexNet.add(Dropout(0.4))

    #3rd Fully Connected Layer
    AlexNet.add(Dense(1000))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    #Add Dropout
    AlexNet.add(Dropout(0.4))

    #Output Layer
    AlexNet.add(Dense(1))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('sigmoid'))
    
    return AlexNet

#Helper Function: Create a Keras prebuilt model
def makePrebuiltModel(inputShape: Tuple[int], modelName:str ='') -> keras.Model:
    """
    Load model by inputing the name to modelName
    Options are "InceptionResNetV2", "MobileNetV2", "ResNet50V2", "DenseNet121", "DenseNet201", and "NASNetLarge"
    """
    input = keras.Input(shape=inputShape, name="Input")
    x=None  
    if modelName=="InceptionResNetV2":
        baseModel = keras.applications.InceptionResNetV2(include_top=False, weights="imagenet", input_shape=inputShape)(input)
    elif modelName=="MobileNetV2":
        baseModel = keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=inputShape)(input)
    elif modelName=="ResNet50V2":
        baseModel = keras.applications.ResNet50V2(include_top=False, weights="imagenet", input_shape=inputShape)(input)
    elif modelName=="DenseNet121":
        baseModel = keras.applications.DenseNet121(include_top=False, weights="imagenet", input_shape=inputShape)(input)
    elif modelName=="DenseNet201":
        baseModel = keras.applications.DenseNet201(include_top=False, weights="imagenet", input_shape=inputShape)(input)
    elif modelName=="NASNetLarge":
        x = tf.keras.layers.experimental.preprocessing.Resizing(height=331, width=331)(input)
        baseModel = keras.applications.NASNetLarge(include_top=False, weights="imagenet", input_shape=(331,331,3))(x)
    else:
        raise("Model Name Not In Recognized Keras Models")
        
    baseModel.trainable = False
    x = layers.Flatten()(baseModel)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    if x != None:
        output=layers.Dense(1, activation="sigmoid")(x)
        return keras.Model(inputs=input, outputs=output, name=modelName)
    else:
        raise("Model Name Not Recognized")
        
#Helper Function: Create a very simple model as a baseline
def makeSimpleNet(inputShape: Tuple[int], modelName:str ='') -> keras.Model:
    input = keras.Input(shape=inputShape, name="Input")
    x = layers.AveragePooling2D(pool_size=(50, 50))(input)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    output=layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs=input, outputs=output, name=modelName)

#Helper Function: Create a mini AlexNet
def makemAlexNet(inputShape: Tuple[int], modelName:str ='') -> keras.Model:
    inputs = keras.Input(shape=inputShape, name="Input")
    
    x = tf.keras.layers.experimental.preprocessing.Resizing(height=224, width=224)(inputs)
    x = layers.Conv2D(filters=16, kernel_size=11, strides=4, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)
    x = layers.Conv2D(filters=20, kernel_size=5, strides=1, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)
    x = layers.Conv2D(filters=30, kernel_size=3, strides=1, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)
    x = layers.Dense(units=48, activation="relu")(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(units=1, activation="sigmoid")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name=modelName)
    return model

#Helper Function: Create a mini DenseNet
def makemDenseNet(inputShape: Tuple[int], modelName:str ='') -> keras.Model:
    def makeDenseBlock(groupCount: int, inputs):
        blockConcats = []
        x = layers.BatchNormalization()(inputs)
        x = layers.Conv2D(filters=16, kernel_size=(1, 1), activation="relu", padding="same")(x)
        x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")(x)
        blockConcats.append(x)
        for count in range(groupCount):
            x = layers.Concatenate()(blockConcats) if len(blockConcats) > 1 else x
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(filters=16, kernel_size=(1, 1), activation="relu", padding="same")(x)
            x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")(x)
            blockConcats.append(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters=16, kernel_size=(1, 1), activation="relu", padding="same")(x)
        x = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        return x
    
    inputs = keras.Input(shape=inputShape, name="Input")
    x = layers.Conv2D(filters=16, kernel_size=(7, 7), strides=(5, 5), activation="relu")(inputs)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    
    x = makeDenseBlock(groupCount=2, inputs=x)
    
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=modelName)

#Helper Function: Create a simple DenseNet
def makeSimpleDenseNet(inputShape: Tuple[int], modelName:str ='') -> keras.Model:
    def makeDenseBlock(groupCount: int, inputs):
        blockConcats = []
        x = layers.BatchNormalization()(inputs)
        x = layers.Conv2D(filters=64, kernel_size=(1, 1), activation="relu", padding="same")(x)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same")(x)
        blockConcats.append(x)
        for count in range(groupCount):
            x = layers.Concatenate()(blockConcats) if len(blockConcats) > 1 else x
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(filters=64, kernel_size=(1, 1), activation="relu", padding="same")(x)
            x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same")(x)
            blockConcats.append(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters=64, kernel_size=(1, 1), activation="relu", padding="same")(x)
        x = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        return x
    
    inputs = keras.Input(shape=inputShape, name="Input")
    x = layers.Conv2D(filters=32, kernel_size=(7, 7), strides=(2, 2), activation="relu")(inputs)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    
    x = makeDenseBlock(groupCount=6, inputs=x)
    
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=modelName)

#Helper Function: Create a simple ResNet
def makeSimpleResNet(inputShape: Tuple[int], modelName:str ='') -> keras.Model:
    """
    Source: https://www.tensorflow.org/guide/keras/functional#a_toy_resnet_model
    """
    inputs = keras.Input(shape=inputShape, name="Input")
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(inputs)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu")(x)
    block_1_output = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(x)

    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same")(block_1_output)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same")(x)
    block_2_output = layers.add([x, block_1_output])

    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same")(block_2_output)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same")(x)
    block_3_output = layers.add([x, block_2_output])

    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu")(block_3_output)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(units=256, activation="relu")(x)
    x = layers.Dense(units=256, activation="relu")(x)
    x = layers.Dense(units=256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=modelName)