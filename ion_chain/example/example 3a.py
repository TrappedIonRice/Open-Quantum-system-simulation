# -*- coding: utf-8 -*-
"""
Compute the time evolution of a 3 ion system constructed to simulate excitation transfer
between 2 sites using Hamiltonian in resonant interaction frame
@author: zhumj
"""
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import ion_chain.operator.spin as spin
import ion_chain.operator.phonon as phon
import ion_chain.transfer.exci_transfer as extrans
from  ion_chain.ising.ion_system import *
#%%
'''
parameters of the system, in this example, we compute the evolution at type 1
resonance at Delta E = 1*delta_rock
'''       
ion_sys = ions() 
ion_sys.delta_ref = 2
ion_sys.N = 3
ion_sys.delta = -100
ion_sys.fr = 50; ion_sys.fb = 50
ion_sys.pcut = np.array([2,2,8]) 
ion_sys.phase = np.pi/2
ion_sys.gamma = [0,0,10] #cool the rocking mode only
ion_sys.list_para() #print parameters
ion_sys.plot_freq()
#%%
'''
define operators for measurement
'''
elist = []
for j in range(3): #phonon population
    eop = tensor(spin.sI(2),phon.up(j,ion_sys.pcut,ion_sys.N)*phon.down(j,ion_sys.pcut,ion_sys.N))
    elist.append(eop) 
oplist = [tensor(spin.sz(2,0),phon.pI(ion_sys.pcut,ion_sys.N)),
          tensor(spin.sz(2,1),phon.pI(ion_sys.pcut,ion_sys.N)),] #spin population
elist = oplist+elist    
#%%    
'''
simulation with 1 mode
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
H0, clist1 = extrans.Htot(J23,(E1-E2)/2,0,V,ion_sys,0) #generate Hamiltonian
rho0 = extrans.rho_ini(ion_sys,True) #initial state
tplot0 = np.arange(0,2,0.01)
times0 =tplot0/tscale
#result = mesolve(H0,rho0,times,clist1,[],progress_bar=True,options=Options(nsteps=10000))
result1 = mesolve(H0,rho0,times0,clist1,elist,progress_bar=True,options=Options(nsteps=100000))
#%% plot spin popluation
pplot1 =  result1.expect[0]
pplot2 =  result1.expect[1]
plt.figure(0)
plt.plot(tplot0,0.5*pplot1+0.5,'-',label=r'$P_{\uparrow\!\!\!\!\downarrow} $')
plt.plot(tplot0,(0.5*pplot2+0.5),'-',label=r'$P_{\downarrow\!\!\!\!\uparrow} $')
#plt.xlim(0,800)    
plt.xlabel(r'$\omega_0t/(2\pi)$')
plt.ylabel(r'$p_{tot}$')
title = r'$\delta_{rock} = -100kHz, \Delta E = $'+str(E1) + r'$kHz , J=$'+str(J23)+r'$kHz$'
plt.title(title)
plt.yticks(fontsize = 13)
plt.xticks(fontsize = 13)
plt.legend(fontsize=10)
plt.grid()   
plt.show()
#%% plot phonon population
plt.figure(1)
phplot1 =  result1.expect[2]
phplot2 = result1.expect[3]
phplot3 =  result1.expect[4]   
plt.plot(times0,phplot1,'r',label='COM')
plt.plot(times0,phplot2,'b--',label = 'tilt')
plt.plot(times0,phplot3,'g--', label = 'rocking')
plt.xlabel(r'$t\delta_0$',fontsize = 14)
plt.ylabel(r'$<a^+a>$',fontsize = 14)
plt.title(title)
plt.yticks(fontsize = 13)
plt.xticks(fontsize = 13)
plt.legend(fontsize=10)
plt.grid()
plt.show()
