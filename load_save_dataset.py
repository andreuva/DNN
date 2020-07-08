import pickle, bz2, glob

datasets = sorted(glob.glob('../data/compresed*'))

for filename in datasets:
	file = bz2.BZ2File(filename, 'rb')
	dict_dataset = pickle.load( file, encoding="latin1" )
	file.close()

	# sfile = bz2.BZ2File('dataset_pickle_bz2.p', 'w')
	sfile = open( "../data/"+filename[18:] , "wb" )
	pickle.dump( dict_dataset , sfile )
	sfile.close()