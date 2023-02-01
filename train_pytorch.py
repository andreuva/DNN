# IMPORT THE MODULES NEEDED
import os, glob, math, random, datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
# import torchvision as tv
import torch.nn as nn
import wandb
import cv2


try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False

## Define the parameters of the network and the save/log/data directories
# Change parameters here...
sizex = 128                 # 128 x
sizey = 128                 # 128 pixels
sizez = 3                   # 3 color channels: RGB
rand_seed = 666             # Seed for random processes
batch_size = 32 * 2         # Size of batches passed to the GPU(s)
nb_epochs = 100             # Number of epochs for training
validation_ratio = 0.2      #Ratio of dataset used as validation
log_dir = "../data/DNN/logs/"        # Where log files will be saved
data_dir = '../data/DNN/data/'       # Where data is stored
resnet_select = 'resnet18'  # Define the desired resnet model
gpu = 0                     # Define the GPU to use

# check if the GPU is available
cuda = torch.cuda.is_available()
device = torch.device(f"cuda:{gpu}" if cuda else "cpu")
print('Checking the GPU availability...')
if cuda:
    print('GPU is available')
    print('Using GPU {}'.format(gpu))
else:
    print('GPU is not available')
    print('Using CPU')
    print(device)


# Ensure log file directory
os.makedirs(log_dir, exist_ok=True)

# Propare output filenames
model_name = resnet_select + '_bs' + str(batch_size)
snapshot_weights = 'models/best_'+model_name+'.hdf5'
last_snapshot_weights = 'models/last_'+model_name+'.hdf5'
json_name = 'models/json_'+model_name+'.json'
trained_model = 'models/trained_'+model_name+'.h5'

# Loss function: Use negative log likelihood for a PDF output
negloglik = lambda y, p_y: -p_y.log_prob(y)

# Translate the last comment from TF to PyTorch
ResNet = torch.hub.load('pytorch/vision:v0.6.0', resnet_select, pretrained=False)
model = nn.Sequential(
    ResNet,
    nn.Flatten(),
    nn.Linear(512, 64),
    nn.LeakyReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 32),
    nn.LeakyReLU(),
    nn.Dropout(0.2),
    nn.Linear(32, 2),
)

# Collect all the .npz files
numpy_datasets = glob.glob(data_dir+'dataset*')[1::2]

# load the data into a dictionary from npz files
data_dict = {}
for dataset in numpy_datasets:
    with np.load(dataset) as data:
        if 'y' in data_dict.keys():
            data_dict['x_nodust'] = np.append(data_dict['x_nodust'], data['x_nodust'], axis=0)
            data_dict['y']        = np.append(data_dict['y'], data['y'], axis=0)
            data_dict['ids']      = np.append(data_dict['ids'], data['id'], axis=0)
        else:
            data_dict['x_nodust'] = data['x_nodust']
            data_dict['y']        = data['y']
            data_dict['ids']       = data['id']

print('Loaded data from ', len(numpy_datasets), ' files')
print('Total number of images: ', len(data_dict['x_nodust']))
data_dict['ids'] = np.array(data_dict['ids'], dtype = str)

# Filter the galaxies that have less than 1e10 solar masses or completely black images
a = np.amax(data_dict['x_nodust'], axis = (1,2,3))
bad_ids = [[galaxy_id,i] for galaxy_id,i in zip(data_dict['ids'],range(len(data_dict['ids']))) if 'e09' in galaxy_id or a[i]<0.5]
bad_ids = np.array(bad_ids)
remove_galaxies = np.array(bad_ids[:,1],dtype=int)

print('there are ', len(bad_ids), ' galaxies to remove')
for key in data_dict.keys():
    data_dict[key] = np.delete(data_dict[key],remove_galaxies, axis=0)
print('Still have ', len(data_dict['ids']), ' galaxies left')

# Define dataset size and buffer size to use the whole dataset
ds_size = len(data_dict['ids'])
buf_size = 2*ds_size

train_size = int(np.floor((1-validation_ratio)*ds_size))
val_size = int(np.ceil(validation_ratio*ds_size))

# Obtain names of unique galaxies in the dataset
ALLuniqueids = data_dict['ids'].copy()
for ii in range(np.size(data_dict['ids'])):
    ALLuniqueids[ii] = data_dict['ids'][ii][:data_dict['ids'][ii].index('e')+3]
uniqueids = np.unique(ALLuniqueids)

train_uniqueids_num = int(np.floor((1-validation_ratio)*np.size(uniqueids)))
val_uniqueids_num = int(np.ceil(validation_ratio*np.size(uniqueids)))

# Generate random indexes to chose train and validation galaxies
random.seed(rand_seed)
randindexes = random.sample(range(np.size(uniqueids)), np.size(uniqueids))

# Get masks to select training and validation samples
train_mask = np.isin(ALLuniqueids, uniqueids[randindexes[:train_uniqueids_num]])
val_mask = np.isin(ALLuniqueids, uniqueids[randindexes[train_uniqueids_num:]])

# computing the normaliced vectors in 3D
# data_dict['y_norm'] = data_dict['y']/np.linalg.norm(data_dict['y'], axis=1)

# resize the images to the desired size
data_dict['x_nodust'] = np.array([cv2.resize(img, (sizex, sizey), interpolation = cv2.INTER_AREA) for img in data_dict['x_nodust']])

# Functions to do the augmentation
"""
# 50% Chance to flip an image left to right. If it does,
# change the x component of angular momentum accordingly
def random_flip_lr(image, label, prob=0.5):

    # define the flipper
    flipper = tv.trasnsforms.RandomHorizontalFlip(p=prob)
    # flip the image
    img = flipper(image)

    # if the image is not flipped, return the same label
    if torch.all(img.eq(image)):
        label_f = label
    else:
        label_f = torch.Tensor([-label[0],label[1]])
    return (img, label_f)

# 50% Chance to flip an image up to down. If it does,
# change the y component of angular momentum accordingly
def random_flip_ud(image, label, prob=0.5):

    # define the flipper
    flipper = tv.trasnsforms.RandomVerticalFlip(p=prob)
    # flip the image
    img = flipper(image)

    # if the image is not flipped, return the same label
    if torch.all(img.eq(image)):
        label_f = label
    else:
        label_f = torch.Tensor([label[0],-label[1]])
    return (img, label_f)

# Rotate the image randomly, return the angle with respect to
# x-axis as new label.
def random_rot(image, label, seed=None):
    
    # Define the number of radians to rotate. 24 possible rotations
    number_of_intervalls = 24
    deg = np.random.uniform(0, 360, 1, dtype=np.float32)
    deg = deg//(360/number_of_intervalls) * (360/number_of_intervalls)

    # Transform images
    img = tv.transforms.functional.rotate(image, deg, interpolation = tv.transforms.functional.InterpolationMode.BILINEAR)

    # Transform J components
    x = np.multiply(np.cos(-deg), label[0]) - np.multiply(np.sin(-deg), label[1])
    y = np.multiply(np.sin(-deg), label[0]) + np.multiply(np.cos(-deg), label[1])

    # Obtain angle and restrict to quadrants 1 and 4
    label_f = np.atan2(y,x)*180/np.pi
    if label_f > 90:
        label_f-=180
    if label_f <= -90:
        label_f += 180

    # Renormalize: from -1 to 1
    label_f = label_f / 90

    return (img, label_f)

# Perform the three previous steps of augmentation
def augment(img, label):
    img_f,label_f = random_flip_lr(img,label)
    img_f,label_f = random_flip_ud(img_f,label_f)
    img_f,label_f = random_rot(img_f,label_f)
    return (img_f, label_f)

 """