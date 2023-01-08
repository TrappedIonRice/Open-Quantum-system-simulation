# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 14:14:49 2022
Varify that the module for anharmonic transfer will reproduce the result for electron transfer
in example 2b 

@author: zhumj
"""
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import Qsim.operator.spin as spin
import Qsim.operator.phonon as phon
import Qsim.ion_chain.transfer.elec_transfer as etrans
import Qsim.ion_chain.ising.ising_ce as isce
import Qsim.ion_chain.transfer.exci_operators as exop
import Qsim.ion_chain.transfer.anharmonic_transfer as an_t
from  Qsim.ion_chain.ion_system import *
#%%
'''
Set up the system as in example 2b
parameters of the system, use the same parameter in quantum regime 
'''    
ion_sys = ions() #construct a two ion system using class ions 
ion_sys.N = 2
ion_sys.delta_ref = 0
ion_sys.delta = -20
ion_sys.Omegax = 0.01*np.abs(ion_sys.delta)
ion_sys.pcut = [10,2]
ion_sys.fr = 70; ion_sys.fb = 70
ion_sys.gamma = 0.05*np.abs(ion_sys.delta)/(2*np.pi)
ion_sys.list_para() #print parameters
deltaE = 5*ion_sys.delta #note site energy difference is negative by definition
tplot = np.arange(0,100,1)
times = tplot*2*np.pi/ion_sys.w0()
elist2 = [tensor(spin.sz(1,0),phon.pI(ion_sys.pcut,2))]
print('g = ',ion_sys.g(),'kHz')
#%%
#solve time evolution for a single energy splitting
H0, clist1 = etrans.Htot(deltaE,ion_sys,False)
rho0 = etrans.rho_ini(ion_sys,False)
print("__________________________________________________________")
print("solving time evolution (2 mode) for deltaE =", deltaE)
result2 = mesolve(H0,rho0,times,clist1,elist2,progress_bar=True,options=Options(nsteps=10000))
rhoee2 = 0.5*result2.expect[0]+0.5
#%% set up the system using module for anharmonic transfer using 3 ions
#set anharmonic terms to 0, use small cutoff for axial df, only com, tilt will be considered
#simulation with complete H, solving time dependent H cost more time
ion_sys.N = 2
ion_sys.coolant = [1]
ion_sys.gamma = [0.05*np.abs(ion_sys.delta)/(2*np.pi),0]
ion_sys.laser_couple = [0]
ion_sys.pcut = [[2],[10,2]]
ion_sys.active_phonon = [[0],[0,1]]
elist3 = [tensor(spin.sz(1,0),exop.p_I(ion_sys))]

ion_sys.list_para()
#%%
rho0 = exop.rho_ini(ion_sys)
Hce, arg0, clist3 = an_t.Htot(deltaE, ion_sys,1,False)
print("__________________________________________________________")
print("solving time evolution using time-dependent H in ordinary frame, for deltaE =", deltaE)
result3 = mesolve(Hce,rho0,times,clist3,elist3,args=arg0,progress_bar=True,options=Options(nsteps=10000))
rhoee3 =  0.5*result3.expect[0]+0.5
#%%    
#plot result    
plt.clf()
plt.plot(tplot,rhoee3,'.',label=r'2 mode ordinary frame')
plt.plot(tplot,rhoee2,label=r'2 mode special frame')
title = r'$\Delta E = $' + str(deltaE/ion_sys.delta)+r'$\delta_{com}$'
plt.xlabel(r'$\omega_0t/(2\pi)$',fontsize = 14)
plt.ylabel(r'$P_{\uparrow}$',fontsize = 14)
plt.title(title)
plt.grid()   
#plt.xlim(0,20)
plt.yticks(fontsize = 13)
plt.xticks(fontsize = 13)
plt.legend()
plt.show()
