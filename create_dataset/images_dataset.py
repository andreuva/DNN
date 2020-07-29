import numpy as np
import pynbody
import pynbody.plot as pp
import pynbody.plot.sph
import sys, os, glob, pickle
import matplotlib.pylab as plt
import gc

names =  sorted(glob.glob('/storage/projects/can43/data/NIHAO/g*'))
num_galax = len(names)
for i in range(num_galax):
	if i==18:
		pass
	elif i==96:
		pass
	else:
		names[i] = names[i][-8:]

np.random.seed(65)
num_proyections = 5

features_dust = []
features_nodust = []
features_names = []
labels = []
done = 0

for simname in names[done:]:
	try:
		galaxy = '/storage/projects/can43/data/NIHAO/'+simname+'/01024/'+simname+'.01024'
		if simname == 'g15784_MW' :
			galaxy = '/storage/projects/can43/data/MAGICC/g15784_MW/1008/g15784.01008'
		
		print('analysis of galaxy: ',simname,'num',done,'of',num_galax)

		# load the snapshot and set to physical units
		s = pynbody.load(galaxy)
		s.physical_units()

		# load the halos and select the main halo (i=1)
		h = s.halos()
		i = 1


		### Make pictures: 
		# center on the largest halo and align the disk
		pynbody.analysis.angmom.faceon(h[i])
		pp.stars.render(s,filename=simname+'.faceon_ivu.png',mag_range=(18,30),#dynamic_range=5.0,
			width=50, with_dust=False, b_scale=1.5)
		pp.stars.render(s,filename=simname+'.faceon_dust_ivu.png',mag_range=(18,30),#dynamic_range=5.0,
			width=50, with_dust=True, b_scale=1.5)

		pynbody.analysis.angmom.sideon(h[i])
		pp.stars.render(s,filename=simname+'.sideon_dust_ivu.png',width=50,mag_range=(18,30),#dynamic_range=5.0
			with_dust=True, b_scale=1.5)
		pp.stars.render(s,filename=simname+'.sideon_ivu.png',width=50,mag_range=(18,30),#dynamic_range=5.0
			with_dust=False, b_scale=1.5)


		anglesx,anglesz,anglesy = np.random.randint(low=0, high=360, size=(3,num_proyections))
		for anglex,anglez,angley in zip(anglesx,anglesz,anglesy):

			pynbody.analysis.angmom.sideon(h[i])
			s.rotate_x(anglex)
			s.rotate_z(anglez)
			s.rotate_y(angley)

			J_vec = pynbody.analysis.angmom.ang_mom_vec(h[i].s)
			Jtot = np.sqrt(((np.multiply(h[i].s['j'].transpose(),h[i].s['mass']).sum(axis=1))**2).sum())
			J_norm = J_vec/Jtot
			print('normaliced angular momentum',round(float(J_norm[0]),2),
				round(float(J_norm[1]),2),round(float(J_norm[2]),2))
			
			im_nodust = pp.stars.render(s,width=50,mag_range=(18,30),#dynamic_range=5.0
				with_dust=False, b_scale=1.5, ret_im=True)
			plt.savefig(simname+'.nodust_feature_sideplus_x'+str(anglex)+'_y'+str(angley)+'_z'+str(anglez)+'.png')
			plt.arrow(0,0,10*float(J_norm[0]),10*float(J_norm[1]), head_width=1.5, head_length=2, fc='g', ec='g')
			plt.savefig(simname+'.nodust_check_ivu_x'+str(anglex)+'_y'+str(angley)+'_z'+str(anglez)+'.png')
			plt.close()

			im_dust = pp.stars.render(s,width=50,mag_range=(18,30),#dynamic_range=5.0
				with_dust=True, b_scale=1.5, ret_im=True)
			plt.savefig(simname+'.dust_feature_sideplus_x'+str(anglex)+'_y'+str(angley)+'_z'+str(anglez)+'.png')
			plt.arrow(0,0,10*float(J_norm[0]),10*float(J_norm[1]), head_width=1.5, head_length=2, fc='g', ec='g')
			plt.savefig(simname+'.dust_check_x'+str(anglex)+'_y'+str(angley)+'_z'+str(anglez)+'.png')
			plt.close()

			features_dust.append(im_dust)
			features_nodust.append(im_nodust)
			features_names.append(simname+'_x'+str(anglex)+'_y'+str(angley)+'_z'+str(anglez))
			labels.append(J_vec)

			gc.collect()
	except:
		print('Problems creating images from galaxy: ',simname)

	done += 1
	if done % 5 == 0:
		# save the labels and features paired
		dataset = {	'features_dust':features_dust,
					'features_nodust':features_nodust,
					'features_names':features_names,
					'labels': labels}

		#sfile = bz2.BZ2File('compresed_dataset_galaxies_pickle_'+str(done)+'.p', 'w')
		sfile = open( 'dataset_galaxies_pickle_'+str(done)+'.p', "wb" )
		pickle.dump( dataset, sfile )
		sfile.close()

		dataset = {}
		features_dust = []
		features_nodust = []
		features_names = []
		labels = []
		gc.collect()

# save the labels and features paired
dataset = {	'features_dust':features_dust,
			'features_nodust':features_nodust,
			'features_names':features_names,
			'labels': labels}

#sfile = bz2.BZ2File('compresed_dataset_galaxies_pickle_'+str(done)+'.p', 'w')
sfile = open( 'dataset_galaxies_pickle_'+str(done)+'.p', "wb" )
pickle.dump( dataset, sfile )
sfile.close()

dataset = {}
features_dust = []
features_nodust = []
features_names = []
labels = []
gc.collect()