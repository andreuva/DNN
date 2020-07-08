import pickle, glob

datasets = sorted(glob.glob('*dataset*'))
global_dataset = {}

for filename in datasets:
	file = open( "dataset_pickle.p", "wb" )
	dict_dataset = pickle.load( file )
	file.close()

	for key in dict_dataset.keys():
		if key in global_dataset:
			global_dataset[key].append(dict_dataset[key])
		else:
			global_dataset[key] = dict_dataset[key]