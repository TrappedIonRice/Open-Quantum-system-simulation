# -*- coding: utf-8 -*-
"""
Compute the time evolution of a 2 ion system contructed to simulate electron transfer 
Reproduce the result in Schlawin et. al. PRXQuantum Paper
https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.010314
@author: zhumj
"""
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import Qsim.operator.spin as spin
import Qsim.operator.phonon as phon
import Qsim.ion_chain.transfer.elec_transfer as etrans
from  Qsim.ion_chain.ion_system import *
import Qsim.ion_chain.transfer.exci_operators as exop
#%%
'''
parameters of the system, use the same parameter in quantum regime 
'''    
ion_sys = ions() #construct a two ion system using class ions
ion_sys.N = 2 
ion_sys.df_laser = 1 #couple to Radial vibrational modes
ion_sys.laser_couple = [0] #laser applied to ion 1
ion_sys.coolant = [1] #ion 2 as coolant
ion_sys.active_phonon = [[0]] #consider only com mode
ion_sys.pcut = [[20]]
ion_sys.delta_ref = 0 #detuning mearsured from com mode
ion_sys.delta = -20 
Omegax = 0.01*np.abs(ion_sys.delta)
ion_sys.fr = 70; ion_sys.fb = 70
ion_sys.gamma = [0.05*np.abs(ion_sys.delta)/(2*np.pi),0] #cool com mode
ion_sys.list_para() #print parameters
deltaE = 5*ion_sys.delta #note site energy difference is negative by definition
#%%  
'''
simulation with 1 mode, reproduce curve C in Fig 3(B)
'''
elist = [exop.spin_measure(ion_sys,0)]
#solve time evolution for a single energy splitting
H0  = etrans.H_res(Omegax,deltaE,ion_sys)
clist1 = exop.c_op(ion_sys,False)
rho0 = exop.rho_thermal(ion_sys)
tplot = np.arange(0,200,0.1)
times = tplot*2*np.pi/(2*np.pi*np.abs(ion_sys.delta))
print("solving time evolution (1 mode) for deltaE =", deltaE)
result = mesolve(H0,rho0,times,clist1,elist,progress_bar=True,options=Options(nsteps=100000))
#%%
#extract ground state population
rhoee1 = result.expect[0]
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
