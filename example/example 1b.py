# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:23:13 2022

@author: zhumj
Compute the time evolution of the ising coulping with a complete Hamiltonian and
compare the result under a pure spin interaction approximation
"""
#%%
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import Qsim.operator.spin as spin
import Qsim.operator.phonon as phon
import Qsim.ion_chain.ising.ising_ps as iscp
import Qsim.ion_chain.ising.ising_c as iscc
import Qsim.operator.spin_phonon as sp_op
from  Qsim.ion_chain.ion_system import *
from scipy import signal
#%%
'''
set parameters of the system
'''    
ion_sys = ions(trap_config={'N': 2, 'fx': 5, 'fz': 0.2}, 
                   phonon_config={'active_phonon': [[0, 1]], 'pcut': [[3,3]]},
                   laser_config={'Omega_eff': 30, 'df_laser': 1, 'laser_couple': [0, 1],
                                 'delta': 100, 'delta_ref': 0, 'phase': 0}
                   )
ion_sys.list_para() #print parameters of the system
Bz = 0 #Effective magnetic field
N = ion_sys.N
#%%
'''
simulation for complete Hamiltonian
'''
#construct Hamiltonian 
Heff,arg0 = iscc.H_ord(Bz,ion_sys) #construct time-dependent H
#construct initial state (initialized as up up)
spin_config = np.array([0,0])
psi1 = sp_op.ini_state(ion_sys,spin_config,[[0,0]],1)
elist1 = [tensor(spin.sz(N,0),sp_op.p_I(ion_sys)),tensor(spin.sz(N,1),sp_op.p_I(ion_sys))]
#solve time dependent SE
times =  np.arange(0,4,10**(-4))
print('______________________________________________________________________')
print('solving for complete Hamiltonian')
result1 = sesolve(Heff,psi1,times,e_ops=elist1,args = arg0,progress_bar=True,options=Options(nsteps=1000))    
#%% 
'''
simulation with a pure spin approximation
'''
psi0 = spin.phiup(N)  
J = iscp.Jt(ion_sys)
elist2 = [spin.sz(N,0),spin.sz(N,1)]
H = iscp.Hps(J,ion_sys,Bz)
print('______________________________________________________________________')
print('solving for pure spin interaction')
result = mesolve(H,psi0,times,e_ops=elist2,progress_bar=True, options=Options(nsteps=1000))
#%%
#plot result
p1 = 0.5*(result.expect[0]+result.expect[1])
p2 = 0.5*(result1.expect[0]+result1.expect[1])
plt.plot(times,p1,label = 'Spin')
plt.plot(times,p2,label = 'Complete')
plt.xlabel(r'$t$ [ms]',fontsize = 14)
title = r'$\delta_{com} = $'+str(ion_sys.delta)+' kHz'
plt.ylabel(r'$<\sigma_{zz}>$',fontsize = 14)
plt.title(title,fontsize = 14)
plt.yticks(fontsize = 14)
plt.xticks(fontsize = 14)
plt.legend(fontsize = 12)
plt.grid()
plt.show()
#%% extract frequency of oscillation for the complete evolution
dlist = ion_sys.detuning/(2*np.pi)
f, Pxx_den = signal.periodogram(p1-p2, 10000)
plt.semilogy(f, Pxx_den,)
label1 = r'$\delta_{com}=$' + str(dlist[0]) + ' kHz'
label2 = '$\delta_{1}=$' + str(np.round(dlist[1])) + ' kHz'
plt.plot([dlist[0],dlist[0]],[0,1],label=label1)
plt.plot([dlist[1],dlist[1]],[0,1],label=label2)
#plt.ylim([1e-7, 1e2])
plt.xlabel('frequency [kHz]',fontsize = 15)
#plt.ylabel(r'PSD kHz',fontsize = 15)
plt.title('Spectrum',fontsize = 15)
plt.ylim(10**(-8),1)
plt.xlim(0,200)
plt.yticks(fontsize = 13)
plt.xticks(fontsize = 13)
plt.legend(fontsize = 12)
plt.grid()
plt.show()