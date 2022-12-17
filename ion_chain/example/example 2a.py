# -*- coding: utf-8 -*-
"""
Compute the time evolution of a 2 ion system contructed to simulate electron transfer 
with 1 mode or 2 modes
Reproduce the result in Schlawin et. al. PRXQuantum Paper
https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.010314
@author: zhumj
"""
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import ion_chain.operator.spin as spin
import ion_chain.operator.phonon as phon
import ion_chain.ising.ising_ps as iscp
import ion_chain.ising.ising_c as iscc
import ion_chain.transfer.elec_transfer as etrans
import ion_chain.ising.ising_ce as isce
from  ion_chain.ising.ion_system import *
#%%
'''
parameters of the system, use the same parameter in quantum regime 
'''    
ion_sys = ions() #construct a two ion system using class ions 
ion_sys.delta_ref = 1 
ion_sys.delta = -20
ion_sys.Omegax = 0.01*np.abs(ion_sys.delta)
ion_sys.fr = 70; ion_sys.fb = 70
ion_sys.gamma = 0.05*np.abs(ion_sys.delta)/(2*np.pi)
ion_sys.list_para() #print parameters
deltaE = 5*ion_sys.delta #note site energy difference is negative by definition
print('g = ',ion_sys.g(),'kHz')
#%%  
'''
simulation with 1 mode, reproduce curve C in Fig 3(B)
'''
ion_sys.pcut = [20]
elist = [tensor(spin.sz(1,0),phon.pI(ion_sys.pcut,1))]
#solve time evolution for a single energy splitting
H0, clist1 = etrans.Htot(deltaE,ion_sys,True)
rho0 = etrans.rho_ini(ion_sys,True)
tplot = np.arange(0,200,0.1)
times = tplot*2*np.pi/ion_sys.w0()
print("solving time evolution (1 mode) for deltaE =", deltaE)
result = mesolve(H0,rho0,times,clist1,elist,progress_bar=True,options=Options(nsteps=100000))
#%%
#extract ground state population
rhoee1 = 0.5*result.expect[0]+0.5
plt.clf()
plt.plot(tplot,rhoee1)
title = r'$\Delta E = $' + str(deltaE/ion_sys.delta)+r'$\delta_{com}$'
plt.xlabel(r'$\omega_0t/(2\pi)$',fontsize = 14)
plt.ylabel(r'$P_{\uparrow}$',fontsize = 14)
plt.title(title)
plt.grid()   
plt.yticks(fontsize = 13)
plt.xticks(fontsize = 13)
plt.show()
