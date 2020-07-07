import pickle, bz2, glob

datasets = sorted(glob.glob('compresed*'))
global_dataset = {}

for filename in datasets:
	file = bz2.BZ2File(filename, 'rb')
	dict_dataset = pickle.load( file, encoding="latin1" )
	file.close()

	for key in dict_dataset.keys():
		if key in global_dataset:
			global_dataset[key].append(dict_dataset[key])
		else:
			global_dataset[key] = dict_dataset[key]

sfile = bz2.BZ2File('dataset_pickle_bz2.p', 'w')
pickle.dump( global_dataset, sfile )
sfile.close()