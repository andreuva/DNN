import pickle, glob
import numpy as np
import os

datasets = sorted(glob.glob('../../data/*pickle*'))
global_dataset = {}

for filename in datasets:
	file = open( filename, "rb" )
	dict_dataset = pickle.load( file )
	file.close()

	np.savez_compressed('../../data/dataset'+filename[31:]+'.npz',
		x_dust = dict_dataset['features_dust'],
		x_nodust = dict_dataset['features_nodust'],
		y = dict_dataset['labels'],
		id = dict_dataset['features_names'])
	print('saved dataset as:' + 'dataset'+filename[31:-2]+'.npz')
	
	os.remove(filename)
	print('removed pickle:',filename)

	#for key in dict_dataset.keys():
	#	if key in global_dataset:
	#		global_dataset[key].append(dict_dataset[key])
	#	else:
	#		global_dataset[key] = dict_dataset[key]
