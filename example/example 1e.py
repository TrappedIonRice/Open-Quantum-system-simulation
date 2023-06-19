# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:23:13 2022

@author: zhumj
Compute the full dynamics induced by a two-body ising coulping Hamiltonian with 
parametric amplificaiton 
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
                   numeric_config={'active_spin': [0, 1],'active_phonon': [[0, 1]], 'pcut': [[20,20]]},
                   )
ion_sys.list_para() #print parameters of the system
laser1 = Laser(config = {'Omega_eff':30,'wavevector':1,'Dk':np.sqrt(2)*2*np.pi / (355*10**(-9)),
                         'laser_couple':[0,1], 'mu':100+1e3*ion_sys.fx,'phase':0})
laser1.list_para()
Bz = 0 #Effective magnetic field
N = ion_sys.N
#%% simulation for time-depedent Hamiltonian with out PA
#construct Hamiltonian 
Heff,arg0 = iscc.H_ord(Bz,ion_sys,laser1) #construct time-dependent H
#construct initial state (initialized as up up)
spin_config = np.array([0,0])
psi1 = sp_op.ini_state(ion_sys,spin_config,[[0,0]],1)
elist1 = [tensor(spin.sz(N,0),sp_op.p_I(ion_sys)),tensor(spin.sz(N,1),sp_op.p_I(ion_sys))]
#solve time dependent SE
times =  np.arange(0,4,10**(-3))
print('______________________________________________________________________')
print('solving for Hamiltonian without PA')
result1 = sesolve(Heff,psi1,times,args = arg0,progress_bar=True,options=Options(nsteps=1000)) 
#%% with PA
delta = 100
#set modulation parameters
ion_sys.update_PM(para_mod_config = {'f_mod':2*(ion_sys.fx*1000+delta) ,'V_mod':0.9,'d_T':200})
ion_sys.list_para() #print parameters of the system
print('scale factor phi')
gcoef = ion_sys.PA_coef(1,0)/(2*np.pi)
phi = ( (delta-gcoef)/(delta+gcoef) )**0.25
#print(phi)
Delta = (gcoef/phi)**2/(4*(100+1e3*ion_sys.fx))
delta1 = np.sqrt(delta**2 - gcoef**2 )
#print('')
#print(Delta/delta1)
print((1/phi)**2 * delta/delta1)
#%%
#construct Hamiltonian 
H_PA_com, arg_Hpa = Isp.H_PA_td(ion_sys)
Heff2 = Heff + H_PA_com; arg2 = arg0 | arg_Hpa
elist1 = [tensor(spin.sz(N,0),sp_op.p_I(ion_sys)),tensor(spin.sz(N,1),sp_op.p_I(ion_sys))]
#solve time dependent SE
print('______________________________________________________________________')
print('solving for Hamiltonian with PA')
result2 = sesolve(Heff2,psi1,times,args = arg2,progress_bar=True,options=Options(nsteps=1000))      
#%%
#plot result
p1 = expect(elist1[0],result1.states)
p2 = expect(elist1[0],result2.states)
plt.plot(times,p1,label = 'no PA')
plt.plot(times,p2,label = 'PA')
plt.xlabel(r'$t$ [ms]',fontsize = 14)
#title = r'$\delta_{com} = $'+str(ion_sys.delta)+' kHz'
plt.ylabel(r'$<\sigma_{zz}>$',fontsize = 14)
#plt.title(title,fontsize = 14)
plt.yticks(fontsize = 14)
plt.xticks(fontsize = 14)
plt.legend(fontsize = 12)
plt.grid()
plt.show()
#%%phonon evolution
#note to construct these phonon operators, any laser object above can be applied
mp_state1 = expect(sp_op.pstate_measure(ion_sys,laser1,19,0),result2.states) 
#pplot_pa = expect(sp_op.phonon_measure(ion_sys,laser1, mindex=0), result2.states)
#pplot = expect(sp_op.phonon_measure(ion_sys,laser1, mindex=0), result1.states)
print('Maximum phonon population of highest com phonon space')
print(np.max(mp_state1))
plt.plot(times,pplot,label = 'no pa')
plt.plot(times,pplot_pa,label = 'with pa')

plt.xlabel(r'$t$ [ms]',fontsize = 14)
#title = r'$\delta_{com} = $'+str(ion_sys.delta)+' kHz'
plt.ylabel(r'$<a^+ a>$',fontsize = 14)
#plt.title(title,fontsize = 14)
plt.yticks(fontsize = 14)
plt.xticks(fontsize = 14)
plt.legend(fontsize = 12)
plt.grid()
plt.show()