#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the needed modules
from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.utils import losses_utils
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os, glob, tqdm


# In[2]:


# define the model fiile, the weights file and the data file
json_name = glob.glob('*model*')[-1]
weights_name = glob.glob('*.h*')[-1]
data_name = glob.glob('*data*')[-1]
output_dir = 'outputs'


# In[3]:


# load json and create model
json_file = open(json_name, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
print("Loaded model from " + json_name)
# load weights into new model
model.load_weights(weights_name)
print("Loaded weights from " + weights_name)
 
# Define the custom loss function from the cosine similarity impleted in tensorflow
class myCosineSimilarity(tf.keras.losses.Loss):
    def __init__(self,
               axis=-1,
               reduction=losses_utils.ReductionV2.AUTO,
               name='myCosineSimilarity'):
        super(myCosineSimilarity, self).__init__(reduction=reduction, name=name)
        self._axis = axis
    
    def call(self, y_true, y_pred):
        cossim = lambda y, p_y: -1*tf.keras.backend.abs(tf.keras.losses.cosine_similarity(y, p_y, axis=self._axis))
        return cossim(y_true, y_pred)
    
# Compile the model specifying the optimazer (sgd) and the custom loss function and other metrics
model.compile(optimizer= tf.keras.optimizers.SGD(),
              loss= myCosineSimilarity(),
              metrics=[tf.keras.losses.cosine_similarity,
                       tf.keras.metrics.mean_absolute_error,])


# In[4]:


#load the data to do the inference
dataset = np.load(data_name)


# In[5]:


# taking samples from the dataset and evaluating the model
images, labels = dataset['x_nodust'][:50], dataset['y'][:50]
outputs = model.predict(images)


# In[6]:


print(outputs[:10])


# In[7]:


# function to save the images with the predictions and labels
def viz(img, pred, id='pred', label=np.array([0,0,0]), verbose=False):
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(img, cmap='gray', vmin=0, vmax=255,origin='lower')
    
    if np.sum(np.abs(label)) > 0: ax.arrow(250,250,100*label[0],100*label[1], head_width=7, head_length=10, fc='g', ec='g')
    ax.arrow(250,250,100*pred[0],100*pred[1], head_width=7, head_length=10, fc='r', ec='r')
    
    if verbose:
        if np.sum(np.abs(label)) > 0: print(f"label (x,y): ({label})")
        print(f"prediction (x,y): ({pred})")
        print(f"saved image with name: {id}.png")
    plt.savefig(str(id)+'.png')
    plt.close()


# In[8]:


os.makedirs(output_dir, exist_ok=True)
print('saving images to :' + output_dir)


# In[9]:


for image, output, label, i in tqdm.tqdm(zip(images, outputs, labels, range(25))):
    viz(image, output, output_dir+'/'+str(i) , label)


# In[ ]:




