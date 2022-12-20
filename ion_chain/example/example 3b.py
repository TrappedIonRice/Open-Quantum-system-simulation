# -*- coding: utf-8 -*-
"""
Compute the time evolution of a 3 ion system contructed to simulate excitation transfer 
between 2 sites in reasonant interaction frame and using time-dependent Hamiltonian in
ordinary interaction.
verify the results are the same.
@author: zhumj
"""
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import ion_chain.operator.spin as spin
import ion_chain.operator.phonon as phon
import ion_chain.transfer.exci_transfer as extrans
import ion_chain.ising.ising_cex as iscex
from  ion_chain.ising.ion_system import *
#%%
'''
parameters of the system, in this example, we compute the evoluation at type 1
reasonance at Delta E = 1*delta_rock
'''       
ion_sys = ions() 
ion_sys.delta_ref = 0
ion_sys.N = 3
ion_sys.delta = -100
ion_sys.fr = 50; ion_sys.fb = 50
ion_sys.pcut = np.array([5,2,2]) 
ion_sys.phase = np.pi/2
ion_sys.gamma = [10,0,0] #cool the rocking mode only
ion_sys.list_para() #print parameters
ion_sys.plot_freq()
#%%
'''
define operators for measurement
'''
oplist = [tensor(spin.sz(2,0),phon.pI(ion_sys.pcut,ion_sys.N)),
          tensor(spin.sz(2,1),phon.pI(ion_sys.pcut,ion_sys.N)),] #spin population
#%%    
'''
parameters of the system, in this example, we compute the evoluation at type 1
reasonance at Delta E = 1*delta_rock
'''  
#ion_sys.pcut = [3,2,2]
#solve time evolution for a single energy splitting
J23 = 1
E1 = 100 #set energy difference as 100kHz (\delta_rock)
E2 = 0
V = 0
print('coupling strength between ion 1 and 2', J23, ' kHz *h')
print('site energy difference ', E1-E2, ' kHz *h')
configuration = 0 #0 for cooling ion on the side
tscale = J23      #use J as time scale
rho0 = extrans.rho_ini(ion_sys,True) #initial state
tplot0 = np.arange(0,1,0.01)
times0 =tplot0/tscale
#%%
print("__________________________________________________________")
print('simulating with H in resonant interaction frame')
H1, clist1 = extrans.Htot(J23,(E1-E2)/2,0,V,ion_sys,0) #generate Hamiltonian
#result = mesolve(H0,rho0,times,clist1,[],progress_bar=True,options=Options(nsteps=10000))
result1 = mesolve(H1,rho0,times0,clist1,oplist,progress_bar=True,options=Options(nsteps=100000))
#%%
'''
Use complete Hamiltonian in ordinary interaction frame 
'''
print("__________________________________________________________")
print('simulating with time-dependent H in ordinary interaction frame')
H2, arg0, clist2 = iscex.Htot(J23,(E1-E2)/2,0,V,ion_sys,0) #generate Hamiltonian
#result = mesolve(H0,rho0,times,clist1,[],progress_bar=True,options=Options(nsteps=10000))
result2 = mesolve(H2,rho0,times0,clist1,oplist,args= arg0,progress_bar=True,options=Options(nsteps=100000))
#%% plot spin popluation
pplot1 =  result1.expect[0]
pplot2 =  result2.expect[0]
plt.figure(0)
plt.plot(tplot0,0.5*pplot1+0.5,'-',label='special frame')
plt.plot(tplot0,0.5*pplot2+0.5,'+',label=r'ordinary frame')   
plt.xlabel(r'$\omega_0t/(2\pi)$')
plt.ylabel(r'$p_{tot}$')
title = r'$\delta_{rock} = -100kHz, \Delta E = $'+str(E1) + r'$kHz , J=$'+str(J23)+r'$kHz$'
plt.title(title)
plt.yticks(fontsize = 13)
plt.xticks(fontsize = 13)
plt.legend(fontsize=12)
plt.grid()   
plt.show()