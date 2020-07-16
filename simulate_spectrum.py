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
    if header is not None:
        #x is in kms
        x=np.divide([header['CRVAL3']+k*header['CDELT3'] for k in range(header['NAXIS3'])],1000)
    else:
        #x is in kms
        x=[vel0+k*delvel for k in np.arange(vellen)]

    if kms == False:
        x=x*1000
    
    return np.array(x)

def gaussian(length,amplitude,width,position):
    '''length is a float or numpy array, lists will throw operand errors'''
    return amplitude*np.exp(-0.5*((length-position)**2/(width/(2*np.sqrt(2*np.log(2))))**2))

def sumgaussians(length, *args): 
    '''comps should be entered in format sumgaussiasn(length,(amp0,width0,pos0),(amp1,width1,pos1)....)
    
    OR
    
    sumgaussiasn(length,*((amp0,width0,pos0),(amp1,width1,pos1)....))'''
    y=0
    for i in range(len(args)): ###previously did range (1,int(len(...))) not sure if that was just an error left over from a previous indexing attempt
        y+=gaussian(length,args[i][0],args[i][1],args[i][2])
    return y

def simulate_comp(fwhm, pos, header=None, ts_0=None, tau_0=None,vel0=-30,delvel=0.1,vellen=600,vel_ax=None):
    '''simulates a gaussian component. 
    vellen and length are redundant and don't interact properly'''
    #creates velocity axis from header
    if header is not None:
        length = make_vel_ax(header=header)
    #otherwise pull a supplied velocity axis
    if vel_ax is not None:
        length=vel_ax
    #otherwise create custom velocity axis, currently uses length instead of vellen
    else: 
        length=make_vel_ax(vel0=vel0,delvel=delvel,vellen=vellen)

    Ts=ts_0
    tau=gaussian(length,tau_0,fwhm,pos)
    Tb=np.multiply(Ts,(1-np.exp(-tau)))

    return Tb, length

def simulate_spec(*comps,tb_noise=0, tau_noise=0,vel0=-30,delvel=0.1,vellen=600,vel_ax=None):
    '''First component has no opacity from other components and is physically first in the LOS. 
    Second component inputted will have the first blocking it and so on.
    Component should have format (fwhm,pos,Ts,Tau)
    Noise is in std devs and hence in the units of the spectrum.
    '''
    #record the input components
    inputcomps=[i for i in comps]
    
    #establish the velocity space if vel_ax is undefined
    if vel_ax is None:
        vel_ax=make_vel_ax(vel0=vel0,delvel=delvel,vellen=vellen)
    
    #define the first component
    comp1, comp1len = simulate_comp(comps[0][0],comps[0][1],ts_0=comps[0][2],tau_0=comps[0][3],
    vel0=vel0,delvel=delvel,vellen=vellen,vel_ax=vel_ax)
    
    
    #establish the spectra
    spectrum=comp1.copy()
    spectrum_no_opac=comp1.copy()
    
    #set the opacity of the LOS to zero
    sumtaus=0
    
    for i in range(1,len(comps)):
        #take the ith component beginning with the second comp
        newcomp, newcomplen = simulate_comp(comps[i][0],comps[i][1],ts_0=comps[i][2],tau_0=comps[i][3],
        vel0=vel0,delvel=delvel,vellen=vellen,vel_ax=vel_ax)
       
        #add to the opacity spectrum the i-1th component
        sumtaus+=gaussian(vel_ax,comps[i-1][3],comps[i-1][0],comps[i-1][1])
        
        #new spectrum = the new component * np.exp(-opacity spectrum for the preceding components)
        spectrum+=newcomp*np.exp(-sumtaus)
        spectrum_no_opac+=newcomp
        
    #add in the last components opacity to make the opacity spectrum complete, also add in noise
    sumtaus += gaussian(vel_ax,comps[-1][3],comps[-1][0],comps[-1][1])
    sumtaus += (np.random.normal(0,tau_noise,len(vel_ax)))
    
    #add in tb_noise
    noise = np.random.normal(0,tb_noise,len(comp1len))
    spectrum += noise
    spectrum_no_opac += noise
    
    return spectrum, spectrum_no_opac, comp1len, sumtaus, inputcomps

def simulate_spec_lmfit(*comps,length,tb_noise=0, tau_noise=0,vel0=-30,delvel=0.5,vellen=600,vel_ax=None, data=None):
    '''First component has no opacity from other components and is physically first in the LOS. 
    Second component inputted will have the first blocking it and so on.
    Component should have format (fwhm,pos,Ts,Tau)
    Noise is in std devs and hence in the units of the spectrum
    '''
    #record the input components
    inputcomps=[i for i in comps]
    
    comps=comps[0]
    #establish the velocity space
    if vel_ax is not None:
        gausslen = vel_ax
    else:
        gausslen=make_vel_ax(vel0=vel0,delvel=delvel,vellen=length)
    #define the first component
    comp1, comp1len = simulate_comp(comps[f'width0'],comps[f'pos0'],ts_0=comps[f'Ts0'],tau_0=comps[f'tau0'],vel_ax=vel_ax)
    

    #establish the spectra
    spectrum=comp1.copy()
    spectrum_no_opac=comp1.copy()
    
    #set the opacity of the LOS to zero
    sumtaus=0
    
    for i in range(1,int(len(comps)/4)): #divide 4 because each of the components has 4 subcomponents
        #take the ith component beginning with the second comp
        newcomp, newcomplen = simulate_comp(comps[f'width{i}'],
        comps[f'pos{i}'],
        ts_0=comps[f'Ts{i}'],
        tau_0=comps[f'tau{i}'],vel_ax=vel_ax)

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

def simulate_spec_kcomp_lmfit(*comps,
length,
tb_noise=0, 
tau_noise=0,
vel0=-30,
delvel=0.5,
vellen=600,
vel_ax=None, 
frac=0,
data=None):
    '''First component has no opacity from other components and is physically first in the LOS. 
    Second component inputted will have the first blocking it and so on.
    Component should have format (fwhm,pos,Ts,Tau)
    Noise is in std devs and hence in the units of the spectrum
    '''
    #record the input components
    inputcomps=[i for i in comps]

    comps=comps[0]
    
    #establish the velocity space
    if vel_ax is not None:
        gausslen = vel_ax
    else:
        gausslen = make_vel_ax(vel0=vel0,delvel=delvel,vellen=length)
    
    #separate out the warm and cold comps to make them easier to iterate through
    coldcomps = {key:val for (key,val) in comps.items() if 'cold' in key}
    warmcomps = {key:val for (key,val) in comps.items() if 'warm' in key}
    
    #################################
    ##first deal with the cold comps
    #################################
    
    #define the first component
    comp1, comp1len = simulate_comp(coldcomps[f'cold_width0'],coldcomps[f'cold_pos0'],ts_0=coldcomps[f'cold_Ts0'],tau_0=coldcomps[f'cold_tau0'],vel_ax=vel_ax)
    
    #establish the spectra
    spectrum=comp1.copy()
    spectrum_no_opac=comp1.copy()
    
    #set the opacity of the LOS to zero
    sumtaus=0
    
    for i in range(1,int(len(coldcomps)/4)): #divide 4 because each of the components has 4 subcomponents
        #take the ith component beginning with the second comp
        newcomp, newcomplen = simulate_comp(coldcomps[f'cold_width{i}'],
        coldcomps[f'cold_pos{i}'],
        ts_0=coldcomps[f'cold_Ts{i}'],
        tau_0=coldcomps[f'cold_tau{i}'],vel_ax=vel_ax)

        #add to the opacity spectrum the i-1th component
        sumtaus+=gaussian(gausslen,coldcomps[f'cold_tau{i-1}'],coldcomps[f'cold_width{i-1}'],coldcomps[f'cold_pos{i-1}'])
        
        #new spectrum = the new component * np.exp(-opacity spectrum for the preceding components)
        spectrum+=newcomp*np.exp(-sumtaus)
        spectrum_no_opac+=newcomp
    
    #add in the last component's opacity to make the opacity spectrum complete, also add in noise
    sumtaus += gaussian(gausslen,coldcomps[f'cold_tau{int(len(coldcomps)/4)-1}'],coldcomps[f'cold_width{int(len(coldcomps)/4-1)}'],coldcomps[f'cold_pos{int(len(coldcomps)/4)-1}'])
    sumtaus += (np.random.normal(0,tau_noise,len(gausslen)))
    
    #################################
    ##now deal with the warm comps
    #################################

    #only proceed if warm components are specified
    if len(warmcomps.keys()) > 0:
        for i in range(0,int(len(warmcomps)/3)):
            spectrum+=(frac+((1-frac)*np.exp(-sumtaus)))*gaussian(gausslen,warmcomps[f'warm_amp{i}'],warmcomps[f'warm_width{i}'],warmcomps[f'warm_pos{i}'])


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

    print('finished'+str(vcacube))
    return vca_array


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

t_0=time.clock()

#write the orderings to the log so that its saved with the data and is retrievable
orderinglog['indexarray']=indexarray
orderinglog['comp_permutations']=comp_permutations


fit_params = Parameters()
for k in range(len(comp_permutations)):
#for k in range(len(comp_permutations)): # loops over each permutation 
    #needs to be sampled the same as the index array lower down if only taking a subset of 
    #the whole set of possible permutations
    for i in range(len(comp_permutations[k])): #loops over each component in a permutation
        #set the lmfit paramaters:
        fit_params.add(f'width{i}', value=comp_permutations[k][i][0], vary=True,
                       min=comp_permutations[k][i][0]-0.1*np.abs(comp_permutations[k][i][0]),
                       max=comp_permutations[k][i][0]+0.1*np.abs(comp_permutations[k][i][0]))
        fit_params.add(f'pos{i}', value=comp_permutations[k][i][1], vary= True,
                       min=comp_permutations[k][i][1]-0.1*np.abs(comp_permutations[k][i][1]),
                       max=comp_permutations[k][i][1]+0.1*np.abs(comp_permutations[k][i][1]))
        fit_params.add(f'Ts{i}', value=comp_permutations[k][i][2], vary=True,
                       min=0.055, #rms noise on emission is 0.055K
                       max=10000)
        fit_params.add(f'tau{i}', value=comp_permutations[k][i][3], vary=False)
    
    ##plot the input spectrum, kcomps and the reconstructed gaussian components
    #plt.plot(xspace,data)
    #plt.plot(xspace,kcomps)
    #for i in range(len(comp_permutations[k])):
    #    plt.plot(xspace,trb.simulate_comp(fit_params[f'width{i}'].value,
    #                                  fit_params[f'pos{i}'].value,
    #                                  ts_0=fit_params[f'Ts{i}'].value,
    #                                  tau_0=fit_params[f'tau{i}'].value,
    #                                  vel_ax=xspace)[0])
    #plt.figure()

    ##plots look fine the problem is after this point
    
    '''feed in the above paramaters into the simulate_spec function, 
    length is fed in as a kwarg ratherc than arg because lmfit (specifically the args=(x,)) 
    means i need to make the length the last parameter but for a gathered parameter *comps
    it is ambiguous as to which variable is the length'''
    out = minimize(trb.simulate_spec_lmfit, fit_params, method='leastsq', args=(x,), kws={'data': data, 'vel_ax': xspace,'length': x})
    #again need to put in length as a kwarg here. take only the first element since that's all we care about here (opacity ordered Tb)
    
    fit = trb.simulate_spec_lmfit(out.params, length=x, vel_ax=xspace)[0] 
    
    
    #display junk
    #report_fit(out, show_correl=True)
    out.params.pretty_print()   

    #don't think r squared is actually applicable to non-linear data but whatevs
    print(f' R-squared = {1-np.sum(np.square(data-fit))/len(data)}')

    #plot the results
    #plt.plot(xspace, data, c='grey', ls=':',label='data')
    #plt.plot(xspace, fit, c='black', ls='dashed',label='fit',lw=1)
    #plt.plot(xspace, data-fit, 'k.',label='residuals')
    #for i in range(len(comp_permutations[k])):
    #    plt.plot(xspace,trb.simulate_comp(out.params[f'width{i}'],
    #                                 out.params[f'pos{i}'],
    #                                 ts_0=out.params[f'Ts{i}'],
    #                                 tau_0=out.params[f'tau{i}'],vel_ax=xspace)[0],
    #             label=f'comp {i}')
    #    print(f'Comp {i} Ts = {out.params.valuesdict()[f"Ts{i}"]}')
    #plt.xlim(-20,20)
    #plt.axhline(y=0)
    #plt.legend()
    #plt.show()
    
    
    #write the outputs and residuals to a dictionary for each permutation calculation
    orderinglog[f'permutation_{k}']=out.params
    orderinglog[f'permutation_{k}_residuals']=data-fit
    
    print(f'Done {k} of {len(comp_permutations)}')
print(time.clock()-t_0)


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