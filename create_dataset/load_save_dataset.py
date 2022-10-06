import pickle, bz2, glob, os
import numpy as np

datasets = sorted(glob.glob('../../data/dataset*'))

for filename in datasets[:2]:
	file = bz2.BZ2File(filename, 'rb')
	dict_dataset = pickle.load( file, encoding="latin1" )
	file.close()
    
    # sfile = bz2.BZ2File('dataset_pickle_bz2.p', 'w')
    file = open( "../data/"+filename[18:] , "wb" )
    pickle.dump( dict_dataset , sfile )
    sfile.close()
