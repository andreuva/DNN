import os
from glob import glob

archives = sorted(glob('g*'))
files = []
galaxies = []
for i in range(len(archives)):
	if not os.path.isdir(archives[i]):
		files.append(archives[i])
		galaxies.append(archives[i][0:8])

galaxies = sorted(list(set(galaxies)))
# print(galaxies)


for galaxy in galaxies:
	# Create target Directory if don't exist
	if not os.path.exists(galaxy):
	    os.mkdir(galaxy)
	    print("Directory " , galaxy ,  " Created ")
	else:    
	    print("Directory " , galaxy ,  " already exists")

	galaxy_files = [file for file in files if galaxy in file]

	for file in galaxy_files:
		os.rename(file,galaxy+'/'+file)
