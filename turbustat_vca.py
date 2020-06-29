#MAKE PPV SIM CUBE
import matplotlib.pyplot as plt
from astropy.io import fits
import turbustat
from turbustat.simulator import make_3dfield, make_ppv
import astropy.units as u
from spectral_cube import SpectralCube
from turbustat.io.sim_tools import create_image_header, create_cube_header
from radio_beam import Beam
import numpy as np
from turbustat.statistics import VCA, PowerSpectrum
from turbustat.statistics.apodizing_kernels import CosineBellWindow, TukeyWindow, HanningWindow, SplitCosineBellWindow
import schwimmbad
import datetime


########################
def do_vca(input):
	#split inputs into the components
	vcacube, chansamps, array_save_loc, fig_save_loc = input
	print('starting'+str(vcacube))	

	#load in the vcacube
	vcacube=SpectralCube.read(vcacube)
	
	#check for only nans in first slice of cube, even though it reads the unmasked 
	# data the data in the fits may be masked in some way before it reaches this point which will cause the vca to fail if there are only nonfinite values in the subcube

	finites=0
	nonfinites=0	
	for checkx in np.arange(0,len(vcacube.unmasked_data[0,:,0])):
		for checky in np.arange(0,len(vcacube.unmasked_data[0,0,:])):
			if np.isfinite(vcacube.unmasked_data[0,checkx,checky])==True:
				finites+=1
			else:
				nonfinites+=1

	#do vca or skip depending on whether its only NaNs
	if finites < 1:
		return 'data is only NaNs/inf'
	else:
		#do full thickness mom0 SPS and add to array first
		#import data and compute moment 0
		moment0=vcacube.moment(order=0)

		#compute SPS, add in distance at some point as parameter
		pspec = PowerSpectrum(moment0)
		pspec.run(verbose=False, xunit=u.pix**-1)
		vca_array=[pspec.slope,len(vcacube[:,0,0]),pspec.slope_err]

		#iterate VCA over fractions of the total width of the PPV vcacube
		#for i in [128,64,32,16,8,4,2,1]:
		for i in chansamps:
			vcacube.allow_huge_operations=True
			downsamp_vcacube = vcacube.downsample_axis(i, axis=0)
			downsamp_vcacube.allow_huge_operations=True
			vca = VCA(downsamp_vcacube)
			vca.run(verbose=True, save_name=f'{fig_save_loc}_thickness{i}.png')
			vca_array=np.vstack((vca_array,[vca.slope,i,vca.slope_err]))

		#save the array for future plotting without recomputing
		np.save(array_save_loc, vca_array)
		return vca_array
	print('finished'+str(vcacube))
#####################

def multiproc_vca(subcube_locs,channels,output_loc,fig_loc,dimensions):
	""" subcube_locs must be a list containing the string preceding the dimensions and subcube details. 
	e.g. everything before '..._7x7_x1_y2.fits' can be a list of multiple prefixes if needed.

	arrayloc=/priv/myrtle1/gaskap/nickill/smc/vca/turbustatoutput/simcube_him_7x7_avatar 

	channels should be input as a list and be factors of the total channel range in decreasing order e.g. [32,16,8,4,2,1] """
	
	with schwimmbad.MultiPool() as pool:
		print('started multi processing')
		print(datetime.datetime.now())

		#create the lists for multiprocessing
		#vcacube=[f'{subcube_locs}_{dimensions}x{dimensions}_x{i}_y{j}.fits' for j in np.arange(0,dimensions) for i in np.arange(0,dimensions)]
		vcacube=[f'{k}_{dimensions}x{dimensions}_x{i}_y{j}.fits' for k in subcube_locs for j in np.arange(0,dimensions) for i in np.arange(0,dimensions)]
		chansamps=[channels for j in np.arange(0,dimensions) for k in subcube_locs for i in np.arange(0,dimensions)]
		#arrayloc=[f'{output_loc}_{dimensions}x{dimensions}_x{i}_y{j}' for j in np.arange(0,dimensions) for i in np.arange(0,dimensions)]
		arrayloc=[f'{k}_{dimensions}x{dimensions}_x{i}_y{j}' for k in output_loc for j in np.arange(0,dimensions) for i in np.arange(0,dimensions)]
		#figloc=[f'{fig_loc}_{dimensions}x{dimensions}_x{i}_y{j}' for j in np.arange(0,dimensions) for i in np.arange(0,dimensions)]
		figloc=[f'{k}_{dimensions}x{dimensions}_x{i}_y{j}' for k in fig_loc for j in np.arange(0,dimensions) for i in np.arange(0,dimensions)]


		inputs=list(zip(vcacube,chansamps,arrayloc,figloc))
		print(f'THESE ARE THE INPUTS FOR MULTIPROCESSING:{inputs}')

		out = list(pool.map(do_vca, inputs))
		print('finished multiprocessing')
		print(datetime.datetime.now())
	print(out)

# # Use multipool - same as multiprocessing
# with schwimmbad.MultiPool() as pool:
#         print('started multi him')
#         print(datetime.datetime.now())

#         dimensions=7
#         vcacube=['/avatar/nickill/smc/grid_cubes/simcube_him_7x7'+'_x'+str(i)+'_y'+str(j)+'.fits' for j in np.arange(0,dimensions) for i in np.arange(0,dimensions)]
#         arrayloc=['/priv/myrtle1/gaskap/nickill/smc/vca/turbustatoutput/simcube_him_7x7_avatar'+'_x'+str(i)+'_y'+str(j) for j in np.arange(0,dimensions) for i in np.arange(0,dimensions)]
#         figloc=['/priv/myrtle1/gaskap/nickill/smc/vca/turbustatoutput/simcube_him_7x7_avatar'+'_x'+str(i)+'_y'+str(j) for j in np.arange(0,dimensions) for i in np.arange(0,dimensions)]


#         inputs=list(zip(vcacube,arrayloc,figloc))
#         print(inputs)

#         out = list(pool.map(do_vca, inputs))
#         print('finished multi him')
#         print(datetime.datetime.now())
# print(out)
