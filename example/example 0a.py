# -*- coding: utf-8 -*-
"""
This example intends to show how to import modules and read the documentation of functions in this package
Please run one line each time in the console
"""
#%%
#import the spin subpackage from ion_chain.operator and call it "spin" for futher usage
import Qsim.operator.spin as spin
#%%
#print the discription of the package and list the name of all functions
print(spin.__doc__)
#%%
#read the summary for all functions in spin
spin.summary()
#%%
#print the documentation for function sx
help(spin.sx)
#%% this part illustrates how to use 'class' 
#import all functions of ion_system.py from ion_chain.ising 
#import Qsim.operator.spin as spin
from  Qsim.ion_chain.ion_system import *
#%% see the documents for ions class
help(ions)
#read document for a specific function in this class
help(ions.alpha)
#%% construct a 2 ion system with default parameters
two_ion_sys = ions()
#list all attributes
print(two_ion_sys.__dict__)

#%% initialize a 3 ion system with laser drive in axial direction and coupled to
#all 3 ions. Consider all 3 axial phonon modes and set phonon space cutoff as 5
three_ion_sys = ions(trap_config={'N': 3, 'fx': 2, 'fz': 1}, 
                   phonon_config={'active_phonon': [[0, 1, 2]], 'pcut': [[5, 5, 5]]},
                   laser_config={'Omega_eff': 10, 'df_laser': 1, 'laser_couple': [0, 1, 2],
                                 'delta': 20, 'delta_ref': 0, 'phase': 0},
                   cooling_config={'gamma': [2.0, 2.0, 2.0], 'coolant': [2]}
                   )
three_ion_sys .list_para()
#%% update radial confining frequency and recompute related parameters
three_ion_sys.fx = 3
three_ion_sys.update_trap()
three_ion_sys .list_para()
#%%print axial tilt mode and corresponding eigenfrequencies
print('axial tilt mode')
print(three_ion_sys.axial_mode[1])
print('axial tilt eigenfrequency [MHz]')
print(three_ion_sys.axial_freq[1])

