#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:44:18 2019

@author: nicholaskillerbysmith
"""

from astropy import coordinates as coord
from astropy import units as u
from astropy.coordinates import Angle
from gala.coordinates import MagellanicStreamNidever08
import numpy as np
import sys
import pandas as pd

#regfile=gaskap_magreg
#magllower=-20
#maglupper=-10
#magblower=5
#magbupper=15
magllower=float(-20)
maglupper=float(-10)
magblower=float(5)
magbupper=float(15)

#define file lcoation
#regfileloc=sys.argv[1]

regfileloc=str("/Users/nicholaskillerbysmith/gaskap_mag.reg")

#define clipping bounds
#magllower=float(sys.argv[2])
#maglupper=float(sys.argv[3])
#magblower=float(sys.argv[4])
#magbupper=float(sys.argv[5])


#this seems to successfully transform the j2000 to mag coordinates
#c = coord.FK5(ra=256*u.deg, dec=9*u.deg)
#ms = c.transform_to(MagellanicStreamNidever08)
#print(ms)

importreg= pd.read_csv(regfileloc, delim_whitespace=True, header=None)
regfile=importreg.to_numpy(dtype='str')

clippedarray=np.zeros(11)
#convert from gal to mag and check elements
for i in range(0,len(regfile)):
    #load in coordinates as j2000, references one of the tile vertices i think,should change to centre where the text is placed
    c = coord.FK5(ra=float(regfile[i,1])*u.deg, dec=float(regfile[i,2])*u.deg)
    #convert to mag coords
    ms = c.transform_to(MagellanicStreamNidever08)
    #take coord elements
    sep = Angle([ms.L,ms.B])
    #output coord elements as decimal strings
    decimal = sep.to_string(decimal=True)
    #clip elements based on specified bounds
    if float(decimal[0]) < maglupper:
        if float(decimal[0]) > magllower:
            if float(decimal[1]) < magbupper:
                if float(decimal[1]) > magblower:
                    clippedarray=np.vstack((clippedarray,regfile[i]))
        
clippedarray=clippedarray[1:,]

np.savetxt('croppedtilefield.reg',clippedarray,fmt='%s')
            

