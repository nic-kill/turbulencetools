import glob
import os
import subprocess,shlex,shutil
import sys
import numpy as np
from astropy.io import fits
from spectral_cube import SpectralCube


def grid_cube(input_cube,splitfactor,save_loc,lowcut='min',highcut='max',overwrite=True):
	'''Parameters
	input_cube=path to file
	splitfactor=number of subcubes per axis
	
	#save_loc=path to file
	#specify source cube location
	'''

	##Find dimensions

	input_cube=SpectralCube.read(input_cube)
	print(input_cube.shape)

	#attempt at putting in errors, doesn't really work. will just print the statement not terminate the programme
	try:
		assert input_cube.ndim == 3
	except AssertionError:
		print('Cube must be 3D')
	
	
	xlen=input_cube.shape[1]
	ylen=input_cube.shape[2]


	xax=[i*xlen/splitfactor for i in np.arange(splitfactor+1)]
	yax=[i*ylen/splitfactor for i in np.arange(splitfactor+1)]

	##################

	##################
	#Make mom0 to overlay regions on and split off subregions
	#make the mom0 to overwrite
	moment0=input_cube.moment(order=0)

	for j in np.arange(0,splitfactor):
		for i in np.arange(0,splitfactor):
			print('starting x'+str(i)+' y'+str(j))
			#overwrite region boundaries with really high values
			moment0.array[:,int(xax[i])]=1e7
			moment0.array[int(yax[j]),:]=1e7
			#split off sub regions
			sub=input_cube.subcube(xlo=int(xax[i]), xhi=int(xax[i+1]), ylo=int(yax[j]), yhi=int(yax[j+1]), zlo=lowcut, zhi=highcut, rest_value=None)
			#sub=input_cube.subcube(xlo=int(xax[i]), xhi=int(xax[i+1]), ylo=int(yax[j]), yhi=int(yax[j+1]), rest_value=None)
			sub.write(f'{save_loc}_{splitfactor}x{splitfactor}_x{i}_y{j}.fits')
			print('done x'+str(i)+' y'+str(j))
	moment0.write(f'{save_loc}_{splitfactor}x{splitfactor}_regionoutlines.fits')
	#################





