# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 20:16:51 2023
Generate list of site energy differences for computation
@author: zhumj
"""
import numpy as np

import Qsim.ion_chain.transfer.multi_core as mcs
#for instance, we want to simulate deltaE from 0-200kHz, we want resolution of
#1kHz near expected peaks 0, 100, 200kHz and 10kHz resolution for the rest of the E space   
peaklist = list(np.arange(0,300,100))    
Elist = mcs.generate_flist(0,200,peaklist,10,1,5)
print('instance 1:')
print(Elist)
print('_________________________________________________')
#if we only want to simulate around a few points, say E=100kHz, E=200kHz with resolution
#1kHz,, use fplist function
peaklist = np.array([100,200])
Elist = mcs.generate_fplist(peaklist,1,5)
print('instance 2:')
print(Elist)
print('_________________________________________________')