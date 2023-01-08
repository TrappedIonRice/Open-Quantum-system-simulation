# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 14:14:49 2022
Compute the time evolution of a 2 ion system contructed to simulate electron transfer 
with 1 mode or 2 modes
Compare the result using 1 mode (PRXpapaer), 2 mode (special interaction frame), time dependent H (ordinary frame)
and test the validity of changing interaction frames. 
To save computation time, cutoff = 10 is used for phonon space, note this will not give the precise time evolution but 
it is sufficient to verify the consistency between the last 2 frames.
cutoff = 20 will reproduce the result in example 2a
@author: zhumj
"""
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import Qsim.operator.spin as spin
import Qsim.operator.phonon as phon
import Qsim.ion_chain.transfer.elec_transfer as etrans
import Qsim.ion_chain.ising.ising_ce as isce
from  Qsim.ion_chain.ising.ion_system import *
import Qsim.ion_chain.transfer.exci_operators as exop
#%%
'''
parameters of the system, use the same parameter in quantum regime 
'''    
ion_sys = ions() #construct a two ion system using class ions 
ion_sys.N = 2 
ion_sys.delta_ref = 0 #detuning mearsured from com mode
ion_sys.delta = -20
ion_sys.df_laser = 1 #couple to Radial vibrational modes
ion_sys.laser_couple = [0] #laser applied to ion 1
ion_sys.coolant = [1] #ion 2 as coolant
Omegax = 0.01*np.abs(ion_sys.delta)
ion_sys.fr = 70; ion_sys.fb = 70
ion_sys.gamma = [0.05*np.abs(ion_sys.delta)/(2*np.pi),0]
ion_sys.list_para() #print parameters
deltaE = 5*ion_sys.delta #note site energy difference is negative by definition
tplot = np.arange(0,100,1)
times = tplot*2*np.pi/(2*np.pi*np.abs(ion_sys.delta))
#%%  
'''
simulation with 1 mode
'''
ion_sys.active_phonon = [[0]] #consider only com mode
ion_sys.pcut = [[10]]
ion_sys.list_para() #print parameters
H0  = etrans.H_res(Omegax,deltaE,ion_sys)
elist1 =[exop.spin_measure(ion_sys,0)]
clist1 = exop.c_op(ion_sys,False)
rho0 = exop.rho_thermal(ion_sys)
print("__________________________________________________________")
print("solving time evolution (1 mode) for deltaE =", deltaE)
result1 = mesolve(H0,rho0,times,clist1,elist1,progress_bar=True,options=Options(nsteps=10000))
rhoee1 = result1.expect[0]
#%%
#simulation with 2 modes, use cutoff 2 for first mode because we are only cooling com mode
ion_sys.gamma = [0.05*np.abs(ion_sys.delta)/(2*np.pi),0]
ion_sys.active_phonon = [[0,1]] 
ion_sys.pcut = [[10,2]]
elist2 = [exop.spin_measure(ion_sys,0)]
ion_sys.list_para() 
#solve time evolution for a single energy splitting
H0  = etrans.H_res(Omegax,deltaE,ion_sys)
clist2 = exop.c_op(ion_sys,False)
rho0 = exop.rho_thermal(ion_sys)
print("__________________________________________________________")
print("solving time evolution (2 mode) for deltaE =", deltaE)
result2 = mesolve(H0,rho0,times,clist2,elist2,progress_bar=True,options=Options(nsteps=10000))
rhoee2 = result2.expect[0]
#%%
#simulation with complete H, solving time dependent H cost more time
ion_sys.active_phonon = [[0,1]] 
ion_sys.pcut = [[10,2]]
ion_sys.gamma = [0.05*np.abs(ion_sys.delta)/(2*np.pi),0]
elist3 = [exop.spin_measure(ion_sys,0)]
ion_sys.list_para()
rho0 = exop.rho_thermal(ion_sys)
Hce, arg0 = etrans.H_ord(Omegax,deltaE,ion_sys)
clist3 = exop.c_op(ion_sys,False)
print("__________________________________________________________")
print("solving time evolution using time-dependent H in ordinary frame, for deltaE =", deltaE)
result3 = mesolve(Hce,rho0,times,clist3,elist3,args=arg0,progress_bar=True,options=Options(nsteps=10000))
rhoee3 =  result3.expect[0]
#%%    
#plot result    
plt.clf()
plt.plot(tplot,rhoee1,'x',label='1 mode')
plt.plot(tplot,rhoee3,'.',label=r'2 mode ordinary frame')
plt.plot(tplot,rhoee2,label=r'2 mode special frame')
title = r'$\Delta E = $' + str(deltaE/ion_sys.delta)+r'$\delta_{com}$'
plt.xlabel(r'$\omega_0t/(2\pi)$',fontsize = 14)
plt.ylabel(r'$P_{\uparrow}$',fontsize = 14)
plt.title(title)
plt.grid()   
plt.xlim(0,20)
plt.yticks(fontsize = 13)
plt.xticks(fontsize = 13)
plt.legend()
plt.show()
