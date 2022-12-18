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
tplot = np.arange(0,100,1)
times = tplot*2*np.pi/ion_sys.w0()
print('g = ',ion_sys.g(),'kHz')
#%%  
'''
simulation with 1 mode
'''
ion_sys.pcut = [10]
elist = [tensor(spin.sz(1,0),phon.pI(ion_sys.pcut,1))]
#solve time evolution for a single energy splitting
H0, clist1 = etrans.Htot(deltaE,ion_sys,True)
rho0 = etrans.rho_ini(ion_sys,True)
print("solving time evolution (1 mode) for deltaE =", deltaE)
result = mesolve(H0,rho0,times,clist1,elist,progress_bar=True,options=Options(nsteps=10000))
rhoee1 = 0.5*result.expect[0]+0.5
#%%
#simulation with 2 modes, use cutoff 2 for first mode because we are only cooling com mode
ion_sys.pcut = [2,10]
elist = [tensor(spin.sz(1,0),phon.pI(ion_sys.pcut,2))]
ion_sys.list_para() 
#solve time evolution for a single energy splitting
H0, clist1 = etrans.Htot(deltaE,ion_sys,False)
rho0 = etrans.rho_ini(ion_sys,False)
print("solving time evolution (2 mode) for deltaE =", deltaE)
result = mesolve(H0,rho0,times,clist1,elist,progress_bar=True,options=Options(nsteps=10000))
rhoee2 = 0.5*result.expect[0]+0.5
#%%
#simulation with complete H, solving time dependent H cost more time
ion_sys.pcut = [2,10]
elist = [tensor(spin.sz(1,0),phon.pI(ion_sys.pcut,2))]
ion_sys.list_para()
H0c = 2*np.pi*(-0.5 * deltaE * tensor(spin.sz(1,0),phon.pI(ion_sys.pcut,ion_sys.N))
       + ion_sys.Omegax *   tensor(spin.sx(1,0),phon.pI(ion_sys.pcut,ion_sys.N))) 
rho0 = etrans.rho_ini(ion_sys,False)
Hce, arg0 = isce.Htot(H0c,ion_sys)
print("solving time evolution (2 mode) for deltaE =", deltaE)
result = mesolve(Hce,rho0,times,clist1,elist,args=arg0,progress_bar=True,options=Options(nsteps=10000))
rhoee3 =  0.5*result.expect[0]+0.5
#%%    
#plot result    
plt.clf()
plt.plot(tplot,rhoee1,'+',label='1 mode')
plt.plot(tplot,rhoee3,'.',label=r'2 mode ordinary frame')
plt.plot(tplot,rhoee2,label=r'2 mode special frame')
title = r'$\Delta E = $' + str(deltaE/ion_sys.delta)+r'$\delta_{com}$'
plt.xlabel(r'$\omega_0t/(2\pi)$',fontsize = 14)
plt.ylabel(r'$P_{\uparrow}$',fontsize = 14)
plt.title(title)
plt.grid()   
plt.yticks(fontsize = 13)
plt.xticks(fontsize = 13)
plt.legend()
plt.show()
