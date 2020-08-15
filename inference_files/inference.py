#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.models import model_from_json
import tensorflow as tf
import numpy as np
import os

json_name = 'model_xy_SGD_filter_steped_lr.json'
weights_name = "trained_xy_SGD_filter_steped_lr.h5"
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
images, labels = dataset['x_nodust'], dataset['y']
outputs = model.predict(images)


# In[5]:


print('predicted vectors for the input images: ',outputs)


# In[6]:


print('making images with labels and predictions')


# In[13]:


# functions to visualice the data 
import matplotlib.pyplot as plt

def viz(img, label, pred=np.array([0,0,0]), save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(img, cmap='gray', vmin=0, vmax=255,origin='lower')
    ax.arrow(250,250,100*label[0],100*label[1], head_width=7, head_length=10, fc='g', ec='g')
    if np.sum(np.abs(pred)) > 0: ax.arrow(250,250,100*pred[0],100*pred[1], head_width=7, head_length=10, fc='r', ec='r')
    if save_path: plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()


# In[14]:


from tqdm import tqdm

for image,label,prediction,i in tqdm(zip(images,labels,outputs,range(len(outputs)))):
    viz(image, label, prediction, save_path='predictions/'+str(100+i)+'.png', show=False)


# In[ ]:




