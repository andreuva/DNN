#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.models import model_from_json
import tensorflow as tf
import numpy as np
import os

json_name = 'model_xy_test250_SGD_cosine.json'
weights_name = "trained_xy_test250_SGD_cosine.h5"
data_name = "dataset_2_30.p.npz"


# In[2]:


# load json and create model
json_file = open(json_name, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(weights_name)
print("Loaded model from disk")
 
cossim = lambda y, p_y: -1*tf.keras.backend.abs(tf.keras.losses.cosine_similarity(y, p_y))
model.compile(optimizer= 'SGD', loss=cossim, metrics=[
                        tf.keras.metrics.mean_squared_error, 
                        tf.keras.metrics.mean_absolute_error, 
                        tf.keras.metrics.mean_absolute_percentage_error])


# In[3]:


#load the data to do the inference
dataset = np.load(data_name)


# In[4]:


# taking samples from the dataset and evaluating the model
images, labels = dataset['x_nodust'][:10], dataset['y'][:10]
outputs = model.predict(images)


# In[5]:


print(outputs)

