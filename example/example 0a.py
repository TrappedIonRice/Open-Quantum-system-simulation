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
#%% this part illustrates how to use class ions 
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
three_ion_sys = ions(trap_config={'N': 3, 'fx': 2, 'fz': 0.1}, 
                   numeric_config={'active_spin':[0,1,2], 'active_phonon': [[0, 1, 2]], 'pcut': [[5, 5, 5]]},
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
#%%this part illustrate how to use class 'laser'
help(Laser)
#%%
#construct a laser drive with default parameters
laser1 = Laser()
#list all attributes
print(laser1 .__dict__)
#%%
#update effective laser frequency [kHz] 
laser1.mu = three_ion_sys.fx *1000 + 10
laser1.list_para()
#%%
import Qsim.ion_chain.interaction.spin_phonon as Isp
#this part is about how to use object method, attributes of ions, laser class 
#compute the real rabi frequency based on the effective rabi frequency set
print('Rabi frequency in 2pi kHz', laser1.Omega(three_ion_sys))
#using Laser, ions class object to compute radial Lamb-Dicke coefficients \eta _ im 
LD_mat = np.zeros([3,3])
for i in range(3):
    for m in range(3):
        LD_mat[m,i] = laser1.eta(three_ion_sys.radial_freq[m])* three_ion_sys.radial_mode[m,i]
        #note for convenience of indexing the order of ion index i and mode index m has been switched
        #or compute using specialized function LD_coeff
        #LD_mat[m,i] = Isp.LD_coef(three_ion_sys,laser1,i,m)
print('Lamb Dicke Coefficients') 
print(LD_mat)
#%%
#plot all eigenfrequencies
#plot_all_freq(three_ion_sys)
#the other way is to call plot_N_freq, this function also works for N>3
plot_N_freq(three_ion_sys)
