# -*- coding: utf-8 -*-
"""
Compute the time evolution of a 2 ion system contructed to simulate electron transfer 
with 1 mode or 2 modes
@author: zhumj
"""
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import ion_chain.operator.spin as spin
import ion_chain.operator.phonon as phon
import ion_chain.ising.ising_ps as iscp
import ion_chain.ising.ising_c as iscc
import ion_chain.ising.etransfer as etrans
from  ion_chain.ising.ion_system import *
#%%
'''
    parameters of the system
'''    
ion_sys = ions() #construct a two ion system using class ions 
#ion_sys.list_para() #list default parameters of the system
#modify the cooling rate and cut off
ion_sys.fr = 6.5; ion_sys.fb = 6.5
ion_sys.pcut = 30
ion_sys.gamma = 0.01*ion_sys.delta
ion_sys.list_para() #print parameters
#%%  
'''
simulation with 1 mode
'''
#solve time evolution for a single energy splitting
deltaE = -2*ion_sys.delta
H0, clist1 = etrans.Htot(deltaE,ion_sys,True)
rho0 = etrans.rho_ini(ion_sys,True)
tplot = np.arange(0,100,0.1)
times = tplot*2*np.pi/ion_sys.w0()
print("solving time evolution (1 mode) for deltaE =", deltaE)
result = mesolve(H0,rho0,times,clist1,[],progress_bar=True,options=Options(nsteps=10000))
#%%
#extract ground state population
rhoee1 = np.array([])
for i in range(len(result.states)):
    esum = 0
    for j in range(ion_sys.pcut):
        esum = esum + np.absolute(result.states[i][j,j])
    rhoee1 = np.append(rhoee1,esum)
plt.clf()
plt.plot(tplot,rhoee1,label=r'$\rho_{ee} 1 mode$')
title = r'$\Delta E = $' + str(deltaE/ion_sys.delta)
plt.xlabel(r'$\omega_0t/(2\pi)$')
plt.ylabel(r'$p_{tot}$')
plt.title(title)
plt.grid()   
plt.legend()    
#%%
#simulation with 2 modes
ion_sys.pcut = 15
ion_sys.list_para() 
#solve time evolution for a single energy splitting
H0, clist1 = etrans.Htot(deltaE,ion_sys,False)
rho0 = etrans.rho_ini(ion_sys,False)
print("solving time evolution (2 mode) for deltaE =", deltaE)
result = mesolve(H0,rho0,times,clist1,[],progress_bar=True,options=Options(nsteps=10000))
#%%
#extract ground state population
rhoee2 = np.array([])
for i in range(len(result.states)):
    esum = 0
    for j in range(ion_sys.pcut**2):
        esum = esum + np.absolute(result.states[i][j,j])
    rhoee2 = np.append(rhoee2,esum)
#%%    
#plot result    
plt.clf()
plt.plot(tplot,rhoee1,'+',label=r'$\rho_{ee} 1 mode$')
plt.plot(tplot,rhoee2,label=r'$\rho_{ee} 2 mode$')
title = r'$\Delta E = $' + str(deltaE/ion_sys.w0())
plt.xlabel(r'$\omega_0t/(2\pi)$')
plt.ylabel(r'$p_{tot}$')
plt.title(title)
plt.grid()   
plt.legend()