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
from numpy import exp, linspace, pi, random, sign, sin
from lmfit import Parameters, minimize
from lmfit.printfuncs import report_fit
import string
from itertools import permutations 



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
        for i in range(0,int(len(warmcomps)/3)): #warm comps are specified by (amp,width,pos) hence divide by 3
            spectrum+=(frac+((1-frac)*np.exp(-sumtaus)))*gaussian(gausslen,warmcomps[f'warm_amp{i}'],warmcomps[f'warm_width{i}'],warmcomps[f'warm_pos{i}'])


    #add in tb_noise
    noise = np.random.normal(0,tb_noise,len(comp1len))
    spectrum += noise
    spectrum_no_opac += noise
    
    if data is None:
        return spectrum, spectrum_no_opac, comp1len, sumtaus, inputcomps
    return spectrum-data



########################
def lmfit_multiproc_wrapper(input):
    '''processes one of the permutations on one processor'''

    #split inputs into the components
    comp_ordering, process_no, warmcomps, input_spec, velocityspace = input
    
    print(f'Starting {process_no}')	

    #identify the original spectrum to fit against, this is in Tb here
    #kcomps=trb.sumgaussians(velocityspace,*warmcomps)
    #input_spec_less_kcomps = input_spec-kcomps

    orderinglog=dict()

    #maybe not needed since we feed the input spec in its original form when we specify kcomps are being assessed
    input_spec_less_kcomps=input_spec-(sumgaussians(velocityspace,*warmcomps))

    #throwaway params that do nothing atm because i suck at coding and don't want to break things
    frac=0
    x=10

    fit_params = Parameters()
    print(comp_ordering)

    #parameterise the cold comps output from the ordering solution
    for i, comp in enumerate(comp_ordering):

        fit_params.add(f'cold_width{i}',
        value=comp_ordering[i][0],
        min=comp_ordering[i][0]-0.1*np.abs(comp_ordering[i][0]),
        max=comp_ordering[i][0]+0.1*np.abs(comp_ordering[i][0]),
        vary=True)

        fit_params.add(f'cold_pos{i}',
        value=comp_ordering[i][1],
        min=comp_ordering[i][1]-0.1*np.abs(comp_ordering[i][1]),
        max=comp_ordering[i][1]+0.1*np.abs(comp_ordering[i][1]),
        vary=True)

        fit_params.add(f'cold_Ts{i}',
        value=comp_ordering[i][2],
        min=0, #Ts must be positive
        vary=True)

        fit_params.add(f'cold_tau{i}',
        value=comp_ordering[i][3], 
        vary=False)

    ##parameterise the warm comps
    # for i, comp in enumerate(warmcomps):
    #    fit_params.add(f'warm_amp{i}', value=comp[0], vary=True,
    #                    min=comp[0]-0.1*np.abs(comp[0]),
    #                    max=comp[0]+0.1*np.abs(comp[0]))
    #    fit_params.add(f'warm_width{i}', value=comp[1], vary=True,
    #                    min=comp[1]-0.1*np.abs(comp[1]),
    #                    max=comp[1]+0.1*np.abs(comp[1]))
    #    fit_params.add(f'warm_pos{i}', value=comp[2], vary=True,
    #                   min=comp[2]-0.1*np.abs(comp[2]),
    #                   max=comp[2]+0.1*np.abs(comp[2]))
    

    '''feed in the above paramaters into the simulate_spec function, 
    length is fed in as a kwarg ratherc than arg because lmfit (specifically the args=(x,)) 
    means i need to make the length the last parameter but for a gathered parameter *comps
    it is ambiguous as to which variable is the length'''

    out = minimize(
        simulate_spec_kcomp_lmfit, 
        fit_params, 
        method='leastsq', 
        args=(x,), 
        kws={'data': input_spec_less_kcomps, 'vel_ax': velocityspace,'length': x,'frac':frac}
        )
    #again need to put in length as a kwarg here. take only the first element since that's all we care about here (opacity ordered Tb)

    fit = simulate_spec_kcomp_lmfit(out.params, length=x, vel_ax=velocityspace)[0]
    
    #write the outputs and residuals to a dictionary for each permutation calculation

    orderinglog[f'permutation_{process_no}_frac_{frac}']=out.params
    orderinglog[f'permutation_{process_no}_frac_{frac}_residuals']=input_spec-fit

    #pickle.dump(orderinglog, open(f'{output_loc}', 'wb'))
    print(f'Finished {process_no}')
    return orderinglog


#####################

def multiproc_permutations(
    velocityspace,
    coldcomps,
    warmcomps,
    input_spec,
    output_loc,
    sampstart=None,
    samp_spacing=1,
    sampend=None):

    """
    velocityspace - array 

    coldcomps - tuple of tuples containing the outputs from the gausspy initial guesses (e.g. comps_all_reconstruct) (
    e.g. (width0,pos0,Ts0,tau0),(width1,pos1,Ts1,tau1)....)

    warmcomps - em_comps_no_match

    datatofit - 
    output_loc - 
    sampstart - int
    samp_spacing - int
    sampend - int

    inputcomps should be a tuple of tuples each contianing four elements 

    need to take the output gausspy component initial guesses , input spectrum to fit to

    distribute a unique permutation and output to each processor with the same input spectrum
    """

    #start clock
    t_0=time.time()

    #permute the ordering of the identified components
    comp_permutations = tuple(permutations(coldcomps))

    #create a number of indices equal to the number of components then permute them. 
    #Index array should now be shuffled in the same way as the components
    indexarray=tuple(permutations(tuple(string.ascii_lowercase[:len(comp_permutations[0])])))
    #make the tuples into arrays so i can use list functions on them, 
    #needs to be sampled the same as the permutations array if only taking a subset of 
    #the whole set of possible permutations
    indexarray=np.array(indexarray)

    #specify subset sampling of all permutations to reduce compute time for trials, 
    #defaults to computing all permutations
    indexarray=indexarray[sampstart:sampend:samp_spacing]
    comp_permutations=comp_permutations[sampstart:sampend:samp_spacing]



    
    with schwimmbad.MultiPool() as pool:
        print('started multi processing')
        print(datetime.datetime.now())

        #create the lists for multiprocessing
        comp_ordering=comp_permutations
        process_no=[i for i in range(len(comp_ordering))]
        warmcomps=[warmcomps for i in range(len(comp_ordering))] #warm comps don't change so add the same values to each cold comp permutation
        input_spec=[input_spec for i in range(len(comp_ordering))]
        velocityspace=[velocityspace for i in range(len(comp_ordering))]

        inputs=list(zip(comp_ordering,process_no,warmcomps,input_spec,velocityspace))
        #print(f'THESE ARE THE INPUTS FOR MULTIPROCESSING:{inputs}')

        out = list(pool.map(lmfit_multiproc_wrapper, inputs))
        print('finished multiprocessing')
        print(datetime.datetime.now())
        
    #print(out)
    pickle.dump(out, open(f'{output_loc}.pickle', 'wb'))
    pickle.dump(indexarray, open(f'{output_loc}_indexarray.pickle', 'wb'))
    pickle.dump(comp_permutations, open(f'{output_loc}_comp_permutations.pickle', 'wb'))

    print(time.time()-t_0)

def weighted_comp_vals(orderinglog,comp_permutations,indexarray):
    '''returns the cold comp (FWHM,pos,Ts,tau) with weighting from HT03 applied based 
    on all permutations of the comp ordering given'''
    mean_order_values=dict()

    #trace a given component through all its permutations
    for comp in string.ascii_lowercase[:len(comp_permutations[0])]:
        print(comp)
        comp_of_interest=np.argwhere(indexarray==comp)
        #collect the component's values from each ordering
        compwidth=[orderinglog[ordering[0]][f'permutation_{ordering[0]}_frac_0'].valuesdict()[f'cold_width{ordering[1]}'] for ordering in comp_of_interest]
        comppos=[orderinglog[ordering[0]][f'permutation_{ordering[0]}_frac_0'].valuesdict()[f'cold_pos{ordering[1]}'] for ordering in comp_of_interest]
        compts=[orderinglog[ordering[0]][f'permutation_{ordering[0]}_frac_0'].valuesdict()[f'cold_Ts{ordering[1]}'] for ordering in comp_of_interest]
        comptau=[orderinglog[ordering[0]][f'permutation_{ordering[0]}_frac_0'].valuesdict()[f'cold_tau{ordering[1]}'] for ordering in comp_of_interest]
        
        #calculate wf as the variance of the residuals, sec 3.5 HT03
        wf=[1/(np.std(orderinglog[ordering[0]][f'permutation_{ordering[0]}_frac_0_residuals'])**2) for ordering in comp_of_interest]

        #apply the weighting to the component values for each ordering
        meanwidth=np.sum(np.multiply(compwidth,wf))/(np.sum(wf))
        meanpos=np.sum(np.multiply(comppos,wf))/(np.sum(wf))
        meants=np.sum(np.multiply(compts,wf))/(np.sum(wf))
        meantau=np.sum(np.multiply(comptau,wf))/(np.sum(wf))

        #take these weighted values and input them as a tuple into a dictionary which collates all the components
        mean_order_values[f'{comp}']=(meanwidth,meanpos,meants,meantau)
        print(f'Mean Ts = {meants}')

    meancomps=tuple((mean_order_values[f'{i}'][0],
                     mean_order_values[f'{i}'][1], 
                     mean_order_values[f'{i}'][2], 
                     mean_order_values[f'{i}'][3]) 
                    for i in mean_order_values.keys())
    return meancomps


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