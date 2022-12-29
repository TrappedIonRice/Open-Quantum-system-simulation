# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 17:28:18 2022
@author: zhumj
Compute basic quantities of the anharmonic coupling terms 
"""

from  ion_chain.ising.ion_system import *
import numpy as np
#%%
ion0 = ions()
ion0.N=3
#verify C is invariant under the exchange of two indexes
print('test with index 1,1,0')
print(ion0.C(1,1,0))
print(ion0.C(1,0,1))
print(ion0.C(0,1,1))
print('_____________________________________________')
#varify that Cmmn = -Cmnn
print('test with index 1,0,0')
print(ion0.C(1,0,0))
print(ion0.C(0,0,1))
print(ion0.C(0,1,0))
print('_____________________________________________')
#%% test tensor D
#compare result with theoretical calculation in Marquet's paper eq(25)
#also varify D is symmetric under exchange of two indexes
#note the python index is different from what is used in the paper
print('a 3 ion system')
ion0.N=3
print('compute D233')
print('theoretical result: ', -3/np.sqrt(2)*(4/5)**(4/3))
print('numeric result:') 
print(ion0.D(1,2,2))
print(ion0.D(2,1,2))
print(ion0.D(2,2,1))
print('_____________________________________________')
print('compute D222')
print('theoretical result: ', -1/np.sqrt(2)*(4/5)**(1/3))
print('numeric result:',ion0.D(1,1,1)) 
print('_____________________________________________')
print('a 2 ion system')
ion0.N=2
print('compute D222')
print('theoretical result: ', -2**(1/6))
print('numeric result:',ion0.D(1,1,1)) 
