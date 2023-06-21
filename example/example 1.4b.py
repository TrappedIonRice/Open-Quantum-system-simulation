# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:23:13 2022

@author: zhumj
Simulate 3body coupling with PA and plot time evolution of wigner function.
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
import copy
import imageio
import Qsim.auxiliay_function.wigner as wg

#%% set ion chain parameters
ion_sys = ions(trap_config={'N': 3, 'fx': 5, 'fz': 1}, 
                   numeric_config={'active_spin': [0, 1, 2],'active_phonon': [[0]], 'pcut': [[40]]},
                   )

#%% set parameters of lasers according to the paper
R1 = 26*1e3*2*np.pi #recoil frequency
#compute Dk
Dk1 = np.sqrt(R1*2*MYb171/h) #effective wavenumber
q = 1.3 #compensation scale parameter
delta = 10 #kHz
laser1 = Laser(config = {'Omega_eff':30,'wavevector':1,'Dk':Dk1,'laser_couple':[0,1,2],
                'mu':2*(delta+1e3*ion_sys.fx),'phase':0})
Omega_r = 10*laser1.eta(ion_sys.fx)
Omega_b = 2*np.sqrt(3)*Omega_r/laser1.eta(ion_sys.fx)
laser1.Omega_eff = Omega_b 
#red sideband
laser2 = copy.copy(laser1)
laser2.Omega_eff = Omega_r 
laser2.mu = -1*(delta+1e3*ion_sys.fx)
#compensation beatnote
#compensation blue sideband
laser3 = copy.copy(laser1)
laser3.Omega_eff = np.sqrt(q)*Omega_b 
laser3.mu = 2*(-q*delta+1e3*ion_sys.fx)
#compensation red sideband
laser4 = copy.copy(laser1)
laser4.Omega_eff = np.sqrt(q)*Omega_r 
laser4.mu = -1*(-q*delta+1e3*ion_sys.fx)
print('________________________________________________')
print('Parameteres of laser 1')
laser1.list_para()
print('________________________________________________')
print('Parameteres of laser 2')
laser2.list_para()
print('________________________________________________')
print('Parameteres of laser 3')
laser3.list_para()
print('________________________________________________')
print('Parameteres of laser 4')
laser4.list_para()
#print(laser1.R/(2*np.pi))
print(Omega_r)
N = 3
#%%com mode simulation
def Tthree(eta0, wr,wb,delta0,q):
    J32 = (1+np.sqrt(1/q))*(eta0**4*wr**2*wb)/(16*delta0**2)
    return np.pi/(6*J32)
eta_com = laser1.eta(ion_sys.fx)/np.sqrt(3)
print('predicted period, [ms]')
print(Tthree(eta_com,laser2.Omega(ion_sys),laser1.Omega(ion_sys),delta*2*np.pi,q))
#%% set PA parameters, set V_mod to 0 for dynamics without PA
ion_sys.update_PM(para_mod_config = {'f_mod':2*(ion_sys.fx*1000+delta) ,'V_mod':6*1e-2,'d_T':200})
ion_sys.list_para() #print parameters of the system
print('Predicted phonon osicllation frequency, [kHz]')
gcoef = ion_sys.PA_coef(1,0)/(2*np.pi)
delta1 = np.sqrt(delta**2 - gcoef**2 )
print(delta1)
#%% construct states and operators 
#construct initial state (initialized as up up)
spin_config = np.array([1,1,1])
psi1 = sp_op.ini_state(ion_sys,spin_config,[[0,0,0]],1)
#rho1 = sp_op.ini_state(ion_sys,spin_config,[[0,0,0]],0
elist_com = [tensor(spin.sz(N,0),sp_op.p_I(ion_sys)),
          tensor(spin.sz(N,1),sp_op.p_I(ion_sys)),
          tensor(spin.sz(N,2),sp_op.p_I(ion_sys))]
#projection operators
up = basis(2,0)
down = basis(2,1)
s1 = (basis(2,0)+basis(2,1))/np.sqrt(2) #+
s2 = (basis(2,0)-basis(2,1))/np.sqrt(2) #-
sup = tensor(up,up,up); sdown = tensor(down,down,down)
splus = tensor(s1,s1,s1) ; sminus = tensor(s2,s2,s2)
proj1=tensor(splus*splus.dag(),sp_op.p_I(ion_sys)) #|+++>
proj2=tensor(sminus*sminus.dag(),sp_op.p_I(ion_sys)) #|--->
proj3 = tensor(sup*sup.dag(),sp_op.p_I(ion_sys))
proj4 = tensor(sdown*sdown.dag(),sp_op.p_I(ion_sys))
#%%construct Hamiltonian 
Htb_com, arg_tb = iscc.H_com_multi(ion_sys,laser2,laser1,laser4,laser3,q) #construct time-dependent H
H_PA_com, arg_Hpa = Isp.H_PA_td(ion_sys)
Heff_com = Htb_com + H_PA_com; arg_com = arg_tb | arg_Hpa
#solve time dependent SE
times =  np.arange(0,1,1e-3)
#solve time dependent SE
print('______________________________________________________________________')
print('solving for complete Hamiltonian')
result2 = sesolve(Heff_com,psi1,times,args = arg_com,progress_bar=True,options=Options(nsteps=10000))     
#result2 = mesolve(Heff2,rho1,times,args = arg2,progress_bar=True,options=Options(nsteps=10000))  
#%%plot spin evolution
p2 = expect(elist_com[0],result2.states)
plt.plot(times,p2,label = 'complete H')
plt.xlabel(r'$t$ [ms]',fontsize = 14)
#title = r'$\delta_{com} = $'+str(ion_sys.delta)+' kHz'
plt.ylabel(r'$<\sigma_{zz}>$',fontsize = 14)
plt.yticks(fontsize = 14)
plt.xticks(fontsize = 14)
plt.legend(fontsize = 12)
plt.grid()
plt.show()
#%% plot phonon evolution
mp_state1 = expect(sp_op.pstate_measure(ion_sys,laser1,39,0),result2.states) 
pplot = expect(sp_op.phonon_measure(ion_sys,laser1, mindex=0), result2.states)
print('Maximum phonon population highest com fock state')
print(np.max(mp_state1))  
plt.plot(times,pplot,label= 'complete H')
plt.xlabel(r'$t$ [ms]',fontsize = 14)
plt.ylabel(r'$<a^+ a>$',fontsize = 14)
plt.yticks(fontsize = 14)
plt.xticks(fontsize = 14)
plt.legend(fontsize = 12)
plt.grid()
plt.show()
#%% generate frames for gif, use |---> as projector
tot_frame = 100
wg.wigner_evol_frames(tot_frame,times,result2, proj2, 3)
#%% create gif using generated frames
output_name = '3body pa -.gif'
wg.wiger_evol_gif(tot_frame, output_name, frame_duration=0.25, remove_frame = True)

