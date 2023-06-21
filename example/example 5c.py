# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 17:28:18 2022
@author: zhumj
Compute basic quantities of the anharmonic coupling terms 
"""

from  Qsim.ion_chain.ion_system import *
import numpy as np
import Qsim.ion_chain.transfer.anharmonic_transfer as ah_t
import matplotlib.pyplot as plt
#%%
fx = 3.5
fz = (20/63)**0.5 * fx
ion0 = ions(trap_config = {'N':3,'fx':fx,'fz':fz})
#verify C is invariant under the exchange of two indexes
print('test with index 1,1,0')
print(ion0.ah_C_tensor[1][1][0])
print(ion0.ah_C_tensor[1][0][1])
print(ion0.ah_C_tensor[0][1][1])
print('_____________________________________________')
#varify that Cmmn = -Cmnn
print('test with index 1,0,0')
print(ion0.ah_C_tensor[1][0][0])
print(ion0.ah_C_tensor[0][0][1])
print(ion0.ah_C_tensor[0][1][0])
print('_____________________________________________')
#%% test tensor D
#compare result with theoretical calculation in Marquet's paper eq(25)
#also varify D is symmetric under exchange of two indexes
#note the python index is different from what is used in the paper
print('a 3 ion system')
print('compute D233')
print('theoretical result: ', -3/np.sqrt(2)*(4/5)**(4/3))
print('numeric result:') 
print(ion0.ah_D_tensor[1][2][2])
print(ion0.ah_D_tensor[2][1][2])
print(ion0.ah_D_tensor[1][2][2])
print('_____________________________________________')
print('compute D222')
print('theoretical result: ', -1/np.sqrt(2)*(4/5)**(1/3))
print('numeric result:',ion0.ah_D_tensor[1][1][1]) 
print('_____________________________________________')
print('a 2 ion system')
ion0.N=2
#recompute parameters
ion0.update_trap()
print('compute D222')
print('theoretical result: ', -2**(1/6))
print('numeric result:',ion0.ah_D_tensor[1][1][1]) 
#%% plot all combinations of non-zero D 
ion0.N=3
#recompute parameters
ion0.update_trap()
ion0.plot_ah_c(non_zero=True,real_freq = False)
#%% Check the result with PRL 119 
#compute coupling frequency for mnp = 332
#set the frequency such that wa = 2wb is satisfied
ion0.N=3
ion0.fx = 3.5
ion0.fz = (20/63)**0.5 * ion0.fx
ion0.update_trap()
print('axial confining frequency ', ion0.fz) 
print('axial confining freq calculated using optimization',ah_t.res_freq2(3.5,0,3,[2,2,1]))
#%%
z0 = (5*qe**2 / (16*np.pi * eps0 * MYb171 * fr_conv(ion0.fz,'MHz')**2))**(1/3)
wa = ion0.axial_freq[1]*fr_conv(1,'MHz')
wb = ion0.radial_freq[2]*fr_conv(1,'MHz')
th_result = 9* fr_conv(ion0.fz,'MHz') * np.sqrt(h/(MYb171*wa*wb**2))/(10*z0)
print('_____________________________________________')
print('theoretical result',th_result)
print('numeric result',ion0.ah_couple([2,2,1]))
#%% plot anharmonic coupling coefficient for mode 332 with fx ranging from 2-4MHz
#fx, fz satisfy the reasonant condition for this mode
anc_plot = np.array([])
fxplot = np.arange(2,4,0.01) 
for fx in fxplot:
    ion0.fx = fx
    ion0.fz = (20/63)**0.5 * fx
    ion0.update_trap(print_text=False)
    anc_plot = np.append(anc_plot, ion0.ah_couple([2,2,1],real_unit=True)*10**3) 
    #multiply fz in Hz to get real coupling in unit of frequency
#%%
plt.plot(fxplot,anc_plot)
plt.xlabel('Radial frequency, [MHz]',fontsize = 13) 
#plt.ylabel(r'Anharmonic Coupling, $[\omega_z]$',fontsize = 13)
plt.ylabel(r'Anharmonic Coupling, Hz',fontsize = 13)
plt.xticks(fontsize = 12)    
plt.yticks(fontsize = 12)   
plt.grid()   

