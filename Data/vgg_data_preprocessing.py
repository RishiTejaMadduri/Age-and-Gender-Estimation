#!/usr/bin/env python
# coding: utf-8

# In[15]:


#Loding Libraries and Dependencies

import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from keras.models import Model
from keras.models import model_from_json, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dropout, Activation
import h5py
import pandas as pd
import pickle


# In[2]:


#Loading the pre-trained weights from .mat model
file2='/home/rishi/Projects/Matroid/vgg networks/vgg_face_matconvnet/vgg_face_matconvnet/data/vgg_face.mat'
data=loadmat(file2)


# In[16]:


#HandWritten Model-For extracting weights
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), name= 'conv1_1'))
model.add(Activation('relu', name='relu1_1'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), name= 'conv1_2'))
model.add(Activation('relu', name='relu1_2'))
model.add(MaxPooling2D((2,2), strides=(2,2), name='pool1'))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), name= 'conv2_1'))
model.add(Activation('relu', name='relu2_1'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), name= 'conv2_2'))
model.add(Activation('relu', name='relu2_2'))
model.add(MaxPooling2D((2,2), strides=(2,2), name='pool2'))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), name= 'conv3_1'))
model.add(Activation('relu', name='relu3_1'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), name= 'conv3_2'))
model.add(Activation('relu', name='relu3_2'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), name= 'conv3_3'))
model.add(Activation('relu', name='relu3_3'))
model.add(MaxPooling2D((2,2), strides=(2,2), name='pool3'))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), name= 'conv4_1'))
model.add(Activation('relu', name='relu4_1'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), name= 'conv4_2'))
model.add(Activation('relu', name='relu4_2'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), name= 'conv4_3'))
model.add(Activation('relu', name='relu4_3'))
model.add(MaxPooling2D((2,2), strides=(2,2), name='pool4'))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), name= 'conv5_1'))
model.add(Activation('relu', name='relu5_1'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), name= 'conv5_2'))
model.add(Activation('relu', name='relu5_2'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), name= 'conv5_3'))
model.add(Activation('relu', name='relu5_3'))
model.add(MaxPooling2D((2,2), strides=(2,2), name='pool5'))
 
model.add(Convolution2D(4096, (7, 7), name= 'fc6'))
model.add(Activation('relu', name='relu6'))
model.add(Dropout(0.5, name='dropout6'))
model.add(Convolution2D(4096, (1, 1), name= 'fc7'))
model.add(Activation('relu', name='relu7'))
model.add(Dropout(0.5, name='dropout7'))
model.add(Convolution2D(2622, (1, 1), name= 'fc8'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Activation('softmax', name= 'softmax'))


# In[17]:


#net consists of layers, weights, biases
net=data['net'][0][0]
ref_model_layers=net['layers']


# In[18]:


#Ensuring that only Convolutional and fc layers are being considered
#for extracting weights. Pooling Layers and Relu will give errors.
for i in range(ref_model_layers.shape[1]):
    ref_model_layer=ref_model_layers[0][i][0][0]['name'][0]
    
    try:
        weights=ref_model_layers[0][i][0][0]['weights'][0][0]
        print(ref_model_layer, ":", weights.shape)
    except:
        print("",end="")


# In[19]:


#Checking if MATLAB files and our vars contain the same layers or not
#If they do, extract weights and biases to store in Keras format.
weights_to_save=[]
bias_to_save=[]
base_model_layer_names = [layer.name for layer in model.layers]
num_of_ref_model_layers = ref_model_layers.shape[1]
for i in range(num_of_ref_model_layers):
    ref_model_layer=ref_model_layers[0][i][0][0]['name'][0]
    if ref_model_layer in base_model_layer_names:
        #we just need to set convolution and fully connected weights
        if ref_model_layer.find("conv") == 0 or ref_model_layer.find("fc") == 0:
            print(i,". ",ref_model_layer)
            base_model_index = base_model_layer_names.index(ref_model_layer)

            weights=ref_model_layers[0][i][0][0]['weights'][0][0]
            bias = ref_model_layers[0][i][0][0]['weights'][0][1]
            weights_to_save.append(weights)
            bias_to_save.append(weights)
            model.layers[base_model_index].set_weights([weights, bias[:,0]])


# In[20]:


file='pkl_weights'
outfile = open(file,'wb')
pickle.dump(weights_to_save,outfile)
outfile.close()

