# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:23:13 2022

@author: zhumj
Compute the full dynamics induced by a two-body ising coulping Hamiltonian and
compare the result with the dynamics under a pure spin interaction approximation
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
import Qsim.ion_chain.interaction.spin_phonon as Isp
from  Qsim.ion_chain.ion_system import *
from scipy import signal
import copy
#%%
'''
set parameters of the system
'''    
ion_sys = ions(trap_config={'N': 2, 'fx': 5, 'fz': 0.2}, 
                   numeric_config={'active_spin': [0, 1],'active_phonon': [[0, 1]], 'pcut': [[3,3]]},
                   )
ion_sys.list_para() #print parameters of the system
laser1 = Laser(config = {'Omega_eff':30,'wavevector':1,'Dk':np.sqrt(2)*2*np.pi / (355*10**(-9)),
                         'laser_couple':[0,1], 'mu':100+1e3*ion_sys.fx,'phase':0})
laser1.list_para()
Bz = 0 #Effective magnetic field
N = ion_sys.N
#%%
'''
simulation for time-depedent Hamiltonian under RWA
'''
#construct Hamiltonian 
Heff,arg0 = iscc.H_ord(Bz,ion_sys,laser1) #construct time-dependent H
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
simulation for complete Hamiltonian after first order power series expansion without RWA
'''
laser_b = copy.copy(laser1) #blue sideband
laser_b.phase = -np.pi/2 
laser_r = copy.copy(laser1) #red sideband
laser_r.mu = -laser_b.mu 
laser_r.phase = -np.pi/2
laser_b.list_para()
laser_r.list_para()

#%%
#construct Hamiltonian 
arg2 = Isp.H_td_argdic_general(ion0 = ion_sys, laser_list=[laser_r,laser_b])
Heff2 = Isp.H_td_multi_drives(ion0 = ion_sys, laser_list=[laser_r,laser_b],
                              second_order=False, rwa=False) 
elist1 = [tensor(spin.sz(N,0),sp_op.p_I(ion_sys)),tensor(spin.sz(N,1),sp_op.p_I(ion_sys))]
#solve time dependent SE
print('______________________________________________________________________')
print('solving for complete Hamiltonian')
result2 = sesolve(Heff2,psi1,times,e_ops=elist1,args = arg2,progress_bar=True,options=Options(nsteps=1000))      
#%% 
'''
simulation with a pure spin approximation
'''
psi0 = spin.spin_state(N,[0,0])  
J = iscp.Jt(ion_sys,laser1)
elist2 = [spin.sz(N,0),spin.sz(N,1)]
H = iscp.Hps(J,ion_sys,Bz)
print('______________________________________________________________________')
print('solving for pure spin interaction')
result = mesolve(H,psi0,times,e_ops=elist2,progress_bar=True, options=Options(nsteps=1000))
#%%
#plot result
p0 = 0.5*(result.expect[0]+result.expect[1])
p1 = 0.5*(result1.expect[0]+result1.expect[1])
p2 = 0.5*(result2.expect[0]+result2.expect[1])
plt.plot(times,p0,label = 'Spin')
plt.plot(times,p2,label = 'Complete')
plt.xlabel(r'$t$ [ms]',fontsize = 14)
#title = r'$\delta_{com} = $'+str(ion_sys.delta)+' kHz'
plt.ylabel(r'$<\sigma_{zz}>$',fontsize = 14)
#plt.title(title,fontsize = 14)
plt.yticks(fontsize = 14)
plt.xticks(fontsize = 14)
plt.legend(fontsize = 12)
plt.grid()
plt.show()
#%% compare results from two time-dependent Hamiltonians
plt.plot(times,p1,'x',label = 'RWA')
plt.plot(times,p2,label = 'Complete')
plt.xlim(0,0.2)
plt.xlabel(r'$t$ [ms]',fontsize = 14)
#title = r'$\delta_{com} = $'+str(ion_sys.delta)+' kHz'
plt.ylabel(r'$<\sigma_{zz}>$',fontsize = 14)
#plt.title(title,fontsize = 14)
plt.yticks(fontsize = 14)
plt.xticks(fontsize = 14)
plt.legend(fontsize = 12)
plt.grid()
plt.show()
#%% extract frequency of oscillation for the complete evolution
dlist = laser1.detuning(ion_sys)/(2*np.pi)
f, Pxx_den = signal.periodogram(p1-p2, 10000)
plt.semilogy(f, Pxx_den,)
label1 = r'$\delta_{com}=$' + str(np.round(dlist[0])) + ' kHz'
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