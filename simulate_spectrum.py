import os, glob, time
import numpy as np
import pickle
from tqdm import tqdm_notebook as tqdm

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.interpolate import interp1d

from astropy.io import fits

import gausspy.gausspy.gp as gp
from gausspyplus.utils.gaussian_functions import gaussian, combined_gaussian

from astropy.io import fits

import astropy.units as u
from spectral_cube import SpectralCube

from radio_beam import Beam
import numpy as np
import schwimmbad
import datetime

def make_vel_ax(vel0=None,delvel=None,vellen=None,header=None,kms=True):
    '''vel0, delvel and vellen values given in km/s. Because its supplied in km/s 
    float values will result in rounding errors in the output. 
    Could change it to be in m/s in the future to alleviate this issue if necessary'''
    if header != None:
        #x is in kms
        x=np.divide([header['CRVAL3']+k*header['CDELT3'] for k in range(header['NAXIS3'])],1000)
    else:
        #x is in kms
        x=[vel0+k*delvel for k in np.arange(vellen)]

    if kms == False:
        x=x*1000
    
    return x

def gaussian(length,amplitude,width,position):
    return amplitude*np.exp(-0.5*((length-position)**2/(width/(2*np.sqrt(2*np.log(2))))**2))

def sumgaussians(length, *args): 
    '''comps should be in format sumgaussiasn(length,(amp0,width0,pos0),(amp1,width1,pos1)....)
    
    OR
    
    sumgaussiasn(length,*((amp0,width0,pos0),(amp1,width1,pos1)....))'''
    y=0
    for i in range(len(args)): ###previously did range (1,int(len(...))) not sure if that was just an error left over from a previous indexing attempt
        y+=gaussian(length,args[i][0],args[i][1],args[i][2])
    return y

def gaussian_lmfit(amplitude,width,position,length):
    model=amplitude*np.exp(-0.5*((length-position)**2/(width/(2*np.sqrt(2*np.log(2))))**2))
    #if data is None:
    #    return model
    return model

def sumgaussians_lmfit(args, x, data=None): 
    y=0
    #print(args)
    for i in range(int(len(args)/3)): ###previously did range (1,int(len(...))) not sure if that was just an error left over from a previous indexing attempt
        y+=gaussian_lmfit(args[f'amp{i}'],args[f'width{i}'],args[f'pos{i}'], x)
    if data is None:
        return y
    else:
        return y - data

def simulate_comp(length, fwhm, pos, header=None, tau_0=None,ts_0=None,vel0=-30,delvel=0.1,vellen=600,vel_ax=None):
    '''simulates a gaussian component when given two of the three above values. vellen and length are redundant and don't interact properly'''
    
    #creates velocity axis from header
    if header is not None:
        length = make_vel_ax(header=header)
    if vel_ax is not None:
        length=vel_ax
    else: #creates custom velocity axis, currently uses length instead of vellen
        length=make_vel_ax(vel0=vel0,delvel=delvel,vellen=length)

    Ts=ts_0
    tau=gaussian(length,tau_0,fwhm,pos)
    Tb=np.multiply(Ts,(1-np.exp(-tau)))

    return Tb, length

def simulate_spec(length,*comps,tb_noise=0, tau_noise=0,vel0=-30,delvel=0.5,vellen=600,vel_ax=None):
    '''First component has no opacity from other components and is physically first in the LOS. 
    Second component inputted will have the first blocking it and so on.
    Component should have format (fwhm,pos,Ts,Tau)
    Noise is in std devs and hence in the units of the spectrum.

    vellen and length are redundant and don't interact properly
    '''
    #record the input components
    inputcomps=[i for i in comps]
    
    #establish the velocity space, currently uses length instead of vellen
    if vel_ax is not None:
        gausslen = vel_ax
    else:
        gausslen=make_vel_ax(vel0=vel0,delvel=delvel,vellen=length)
    
    #define the first component
    comp1, comp1len = simulate_comp(length,comps[0][0],comps[0][1],ts_0=comps[0][2],tau_0=comps[0][3],
    vel0=vel0,delvel=delvel,vellen=length,vel_ax=vel_ax)
    
    
    #establish the spectra
    spectrum=comp1.copy()
    spectrum_no_opac=comp1.copy()
    
    #set the opacity of the LOS to zero
    sumtaus=0
    
    for i in range(1,len(comps)):
        #take the ith component beginning with the second comp
        newcomp, newcomplen = simulate_comp(length,comps[i][0],comps[i][1],ts_0=comps[i][2],tau_0=comps[i][3],
        vel0=vel0,delvel=delvel,vellen=length,vel_ax=vel_ax)
       
        #add to the opacity spectrum the i-1th component
        sumtaus+=gaussian(gausslen,comps[i-1][3],comps[i-1][0],comps[i-1][1])
        
        #new spectrum = the new component * np.exp(-opacity spectrum for the preceding components)
        spectrum+=newcomp*np.exp(-sumtaus)
        spectrum_no_opac+=newcomp
        
    #add in the last components opacity to make the opacity spectrum complete, also add in noise
    sumtaus += gaussian(gausslen,comps[-1][3],comps[-1][0],comps[-1][1])
    sumtaus += (np.random.normal(0,tau_noise,len(gausslen)))
    
    #add in tb_noise
    noise = np.random.normal(0,tb_noise,len(comp1len))
    spectrum += noise
    spectrum_no_opac += noise
    
    return spectrum, spectrum_no_opac, comp1len, sumtaus, inputcomps

def simulate_spec_lmfit(*comps,length,tb_noise=0, tau_noise=0,vmin=-30,vmax=30, data=None):
    '''First component has no opacity from other components and is physically first in the LOS. 
    Second component inputted will have the first blocking it and so on.
    Component should have format (fwhm,pos,Ts,Tau)
    Noise is in std devs and hence in the units of the spectrum
    '''
    #record the input components
    inputcomps=[i for i in comps]
    
    comps=comps[0]
    #establish the velocity space
    gausslen=np.linspace(vmin,vmax,length)
    
    #print(f'THESE ARE ALL COMPS:{comps}')
    #print(f'THESE ARE COMPS[0]:{comps[0]}')
    #print(f'THESE ARE ALL COMPS:{inputcomps}')
    #print(f'THESE ARE WIDTH0:{comps["width0"]}')
    #define the first component
    comp1, comp1len = simulate_comp(length,
    comps[f'width0'],
    comps[f'pos0'],
    ts_0=comps[f'Ts0'],
    tau_0=comps[f'tau0'])
    

    #establish the spectra
    spectrum=comp1.copy()
    spectrum_no_opac=comp1.copy()
    
    #set the opacity of the LOS to zero
    sumtaus=0
    
    for i in range(1,int(len(comps)/4)): #divide 4 because each of the components has 4 subcomponents
        #take the ith component beginning with the second comp
        newcomp, newcomplen = simulate_comp(length,
        comps[f'width{i}'],
        comps[f'pos{i}'],
        ts_0=comps[f'Ts{i}'],
        tau_0=comps[f'tau{i}'])

        #add to the opacity spectrum the i-1th component
        sumtaus+=gaussian(gausslen,comps[f'tau{i-1}'],comps[f'width{i-1}'],comps[f'pos{i-1}'])
        
        #new spectrum = the new component * np.exp(-opacity spectrum for the preceding components)
        spectrum+=newcomp*np.exp(-sumtaus)
        spectrum_no_opac+=newcomp
        
    #add in the last components opacity to make the opacity spectrum complete, also add in noise
    sumtaus += gaussian(gausslen,comps[f'tau{int(len(comps)/4)-1}'],comps[f'width{int(len(comps)/4-1)}'],comps[f'pos{int(len(comps)/4)-1}'])
    sumtaus += (np.random.normal(0,tau_noise,len(gausslen)))
    
    #add in tb_noise
    noise = np.random.normal(0,tb_noise,len(comp1len))
    spectrum += noise
    spectrum_no_opac += noise
    
    if data is None:
        return spectrum, spectrum_no_opac, comp1len, sumtaus, inputcomps
    return spectrum-data





########################
def process_ordering(input):
	#split inputs into the components
	vcacube, chansamps, array_save_loc, fig_save_loc = input
	print('starting'+str(vcacube))	

	#load in the vcacube
	vcacube=SpectralCube.read(vcacube)

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

def multiproc_permutations(subcube_locs,channels,output_loc,fig_loc,dimensions):
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
