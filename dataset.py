import numpy as np
import torch
import glob
import cv2
from sklearn.preprocessing import normalize

class galaxy_images(torch.utils.data.Dataset):
    def __init__(self, data_path, train=True, size_x=128, size_y=128, train_split=0.85, rand_seed=666):
        # Collect all the .npz files
        self.files_datasets = glob.glob(data_path+'dataset*')[1::2]

        # load the data
        for dataset in self.files_datasets:
            with np.load(dataset) as data:
                if hasattr(self, 'y'):
                    self.x_nodust = np.append(self.x_nodust, data['x_nodust'], axis=0)
                    self.y        = np.append(self.y, data['y'], axis=0)
                    self.ids      = np.append(self.ids, data['id'], axis=0)
                else:
                    self.x_nodust = data['x_nodust']
                    self.y        = data['y']
                    self.ids      = data['id']
        self.ids = np.array(self.ids, dtype=str)

        print('Loaded data from ', len(self.files_datasets), ' files')
        print('Total number of images: ', len(self.y))

        # Filter the galaxies that have less than 1e10 solar masses or completely black images
        a = np.amax(self.x_nodust, axis = (1,2,3))
        bad_ids = np.array([[galaxy_id,i] for galaxy_id,i in zip(self.ids, range(len(self.ids))) if 'e09' in galaxy_id or a[i]<0.5])
        remove_galaxies = np.array(bad_ids[:,1], dtype=int)

        print('there are ', len(bad_ids), ' galaxies to remove')
        for key in ['x_nodust', 'y', 'ids']:
            setattr(self, key, np.delete(getattr(self,key), remove_galaxies, axis=0))

        print('Still have ', len(self.ids), ' galaxies left')

        # Define dataset size
        if not train:
            self.size = int(np.floor((1-train_split)*len(self.ids)))
        else:
            self.size = int(np.ceil(train_split*len(self.ids)))

        # Obtain names of unique galaxies in the dataset
        ALLuniqueids = self.ids.copy()
        for ii in range(np.size(self.ids)):
            ALLuniqueids[ii] = self.ids[ii][:self.ids[ii].index('e')+3]
        uniqueids = np.unique(ALLuniqueids)

        if not train:
            self.uniqueids_num = int(np.floor((1-train_split)*np.size(uniqueids)))
        else:
            self.uniqueids_num = int(np.ceil(train_split*np.size(uniqueids)))

        # Generate random indexes to chose train and validation galaxies
        np.random.seed(rand_seed)
        randindexes = np.random.sample(range(np.size(uniqueids)), np.size(uniqueids))

        # Get masks to select training and validation samples
        if train:
            self.mask = np.isin(ALLuniqueids, uniqueids[randindexes[:self.uniqueids_num]])
        else:
            self.mask = np.isin(ALLuniqueids, uniqueids[randindexes[-self.uniqueids_num:]])

        # computing the normaliced vectors in 3D
        self.y_norm = normalize(self.y, axis=1)

        # resize the images to the desired size
        self.x_nodust = np.array([cv2.resize(img, (size_x, size_y), interpolation=cv2.INTER_AREA) for img in self.x_nodust])


    def __getitem__(self, index):
        return self.x_nodust[index], self.y_norm[index]
    
    def __call__(self, index):
        return self.x_nodust[index], self.y_norm[index], self.ids[index]

    def __len__(self):
        return self.size
