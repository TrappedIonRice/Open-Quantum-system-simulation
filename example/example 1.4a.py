# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:23:13 2022

@author: zhumj
Simulate 2body coupling with PA and plot time evolution of wigner function.
"""
#%%
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import Qsim.operator.spin as spin
import Qsim.operator.phonon as phon
import Qsim.ion_chain.ising.ising_ps as iscp
import Qsim.ion_chain.ising.ising_c as iscc
import Qsim.operator.spin_phonon as sp_op
import Qsim.ion_chain.interaction.spin_phonon as Isp
from  Qsim.ion_chain.ion_system import *
from scipy import signal
import Qsim.auxiliay_function.wigner as wg
import os
#%% set parameters of ion chain and laser drives
delta = 25
ion_sys = ions(trap_config={'N': 2, 'fx': 3, 'fz': 1}, 
                   numeric_config={'active_spin': [0, 1],'active_phonon': [[0,1]], 'pcut': [[50,50]]},
                   )
ion_sys.list_para() #print parameters of the system
laser1 = Laser(config = {'Omega_eff':30,'wavevector':1,'Dk':np.sqrt(2)*2*np.pi / (355*10**(-9)),
                         'laser_couple':[0,1], 'mu':delta+1e3*ion_sys.fx,'phase':0})
laser1.list_para()
Bz = 0 #Effective magnetic field
N = ion_sys.N
#%% construct states and operators 
spin_config = np.array([0,0])
psi1 = sp_op.ini_state(ion_sys,spin_config,[[0,0]],1)
rho1 = sp_op.ini_state(ion_sys,spin_config,[[0,0]],0)
elist2 = [tensor(spin.sz(N,0),sp_op.p_I(ion_sys)),tensor(spin.sz(N,1),sp_op.p_I(ion_sys))]
#projection operators
up = basis(2,0)
down = basis(2,1)
s1 = (basis(2,0)+basis(2,1))/np.sqrt(2) #|++>
s2 = (basis(2,0)-basis(2,1))/np.sqrt(2) #|-->
splus = tensor(s1,s1) ; sminus = tensor(s2,s2)
proj1=tensor(splus*splus.dag(),sp_op.p_I(ion_sys))
proj2=tensor(sminus*sminus.dag(),sp_op.p_I(ion_sys))
#%% set PA parameters, set V_mod to 0 for dynamics without PA
ion_sys.update_PM(para_mod_config = {'f_mod':2*(ion_sys.fx*1000+delta) ,'V_mod':0.1,'d_T':200})
ion_sys.list_para() #print parameters of the system
print('Predicted phonon osicllation frequency, [kHz]')
gcoef = ion_sys.PA_coef(1,0)/(2*np.pi)
delta1 = np.sqrt(delta**2 - gcoef**2 )
print(delta1)
#%% Construct Hamiltonain
times =  np.arange(0,0.1,10**(-4))
Heff,arg0 = iscc.H_ord(Bz,ion_sys,laser1)
H_PA_com, arg_Hpa = Isp.H_PA_td(ion_sys)
Heff2 = Heff + H_PA_com; arg2 = arg0 | arg_Hpa
#solve time dependent SE
print('______________________________________________________________________')
print('solving for complete Hamiltonian')
result2 = sesolve(Heff2,psi1,times,args = arg2,progress_bar=True,options=Options(nsteps=10000))
#result2 = mesolve(Heff2,rho1,times,args = arg2,progress_bar=True,options=Options(nsteps=10000))     
#%% plot spin dynamics
#p1 = expect(elist1[0],result1.states)
#plt.plot(times,p1,label = 'pure spin')
p2 = expect(elist2[0],result2.states)
plt.plot(times,p2,label = 'complete H')
plt.xlabel(r'$t$ [ms]',fontsize = 14)
plt.ylabel(r'$<\sigma_{zz}>$',fontsize = 14)
plt.yticks(fontsize = 14)
plt.xticks(fontsize = 14)
plt.legend(fontsize = 12)
plt.grid()
plt.show()
#%% plot phonon dynamics
mp_state1 = expect(sp_op.pstate_measure(ion_sys,laser1,49,0),result2.states) 
pplot = expect(sp_op.phonon_measure(ion_sys,laser1, mindex=0), result2.states)
print('Maximum phonon population highest com fock state')
print(np.max(mp_state1))
plt.plot(times,pplot,label = 'complete H')
plt.xlabel(r'$t$ [ms]',fontsize = 14)
plt.ylabel(r'$<a^+ a>$',fontsize = 14)
plt.yticks(fontsize = 14)
plt.xticks(fontsize = 14)
plt.legend(fontsize = 12)
plt.grid()
plt.show()

#%% generate frames for gif, use |--> as projector plot
tot_frame = 25
wg.wigner_evol_frames(tot_frame,times,result2, proj2, 2)
#%% create gif using generated frames
wg.wiger_evol_gif(tot_frame,'2body_PA + .gif',frame_duration=0.5,remove_frame = True)