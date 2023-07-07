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
import Qsim.operator.spin_phonon as sp_op
import Qsim.ion_chain.interaction.dissipation as disp
#%%
'''
parameters of the system, use the same parameter in quantum regime 
'''    
delta = 20 #detuning from com mode
ion_sys = ions(trap_config = {'N':2,'fx':2,'fz':1}) #construct a two ion system using class ions
ion_sys.coolant = [1] #ion 2 as coolant
ion_sys.active_phonon = [[0]] #consider only com mode
ion_sys.gamma = [0.05*np.abs(delta)/(2*np.pi),0] #cool com mode
ion_sys.active_spin = [0]
ion_sys.pcut = [[20]]
ion_sys.update_all(print_text = False)
ion_sys.list_para() #print parameters
#%%
laser1 = Laser()
laser1.Omega_eff = 70
laser1.wavevector = 1 
laser1.laser_couple = [0]
laser1.mu = 1e3*ion_sys.fx-delta
laser1.list_para()
#%%
Omegax = 0.01*np.abs(delta)
deltaE = -5*np.abs(delta) #note site energy difference is negative by definition  
'''
simulation with 1 mode, reproduce curve C in Fig 3(B)
'''
elist = [sp_op.spin_measure(ion_sys,[0,1])]
#solve time evolution for a single energy splitting
H0  = etrans.H_res(Omegax,deltaE,ion_sys,laser1)
clist1 = disp.cooling(ion_sys,[0.01],False)
rho0 = sp_op.rho_thermal(ion_sys,nbar_list=[[0.01]],s_state=[0],ket = False)
tplot = np.arange(0,200,0.1)
times = tplot*2*np.pi/(2*np.pi*np.abs(delta))
print("solving time evolution (1 mode) for deltaE =", deltaE)
result = mesolve(H0,rho0,times,clist1,elist,progress_bar=True,options=Options(nsteps=100000))
#%%
#extract ground state population
rhoee1 = result.expect[0]
plt.clf()
plt.plot(tplot,rhoee1)
title = r'$\Delta E = $' + str(deltaE/delta)+r'$\delta_{com}$'
plt.xlabel(r'$\omega_0t/(2\pi)$',fontsize = 14)
plt.ylabel(r'$P_{\uparrow}$',fontsize = 14)
plt.title(title)
plt.grid()   
plt.yticks(fontsize = 13)
plt.xticks(fontsize = 13)
plt.show()
