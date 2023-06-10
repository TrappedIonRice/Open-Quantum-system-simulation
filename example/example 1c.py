# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:23:13 2022

@author: zhumj
Compute the dynamics induced by 3 body coupling Hamiltonian, reproduce plots 
2(b), 5(a), 6(a), 6(d). The Hamiltonian is constructed by first computing the complete
second-order power series expansion of eq(3) and filter out lower frequency terms
using a designated critrion. This method would take more computation time as the Hamiltonian
has to be separated into many indivdual terms for applying RWA automatically. 
"""
#%%
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import Qsim.operator.spin as spin
import Qsim.operator.spin_phonon as sp_op
import Qsim.ion_chain.interaction.spin_phonon as Isp
from  Qsim.ion_chain.ion_system import *
from scipy import signal
import copy
plt.rcParams['figure.dpi']= 200
#%% set parameters of ion chain
ion_sys = ions(trap_config={'N': 3, 'fx': 5, 'fz': 1}, 
                   numeric_config={'active_spin': [0, 1, 2],'active_phonon': [[0]], 'pcut': [[6]]},
                   )
ion_sys.list_para() #print parameters of the system
#%% set parameters of lasers according to the paper
R1 = 26*1e3*2*np.pi #recoil frequency
#compute Dk
Dk1 = np.sqrt(R1*2*MYb171/h) #effective wavenumber
q = 1.3 #compensation scale parameter
delta = 2 #kHz

#symmetric beatnote
#blue sideband 1
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
print(laser2.Omega(ion_sys)/(2*np.pi))
N = 3
#%%Reproduce plot 2(b), only consider com mode
def Tthree(eta0, wr,wb,delta0,q):
    J32 = (1+np.sqrt(1/q))*(eta0**4*wr**2*wb)/(16*delta0**2)
    return np.pi/(6*J32)
eta_com = laser1.eta(ion_sys.fx)/np.sqrt(3)
print('predicted frequency')
print(Tthree(eta_com,laser2.Omega(ion_sys),laser1.Omega(ion_sys),delta*2*np.pi,q))
#compute rwa critirion
delta_omega = (np.max(ion_sys.radial_freq)- np.min(ion_sys.radial_freq))*1000
rwa_fc = fr_conv(2*q*(delta_omega+delta+10),'hz') 
print('RWA critrion frequency: '+ str(np.round(rwa_fc/(2*np.pi),2))+' [kHz]')
#10 is an additional factor to make sure all frequencies are included
#%% construct Hamiltonian 
arg_com = Isp.H_td_argdic_general(ion0 = ion_sys, laser_list=[laser1,laser2,laser3,laser4])
Heff_com = Isp.H_td_multi_drives(ion0 = ion_sys, laser_list=[laser1,laser2,laser3,laser4],second_order=True,
                              rwa=True,arg_dic=arg_com,f_crit=rwa_fc ) #construct time-dependent H
#print('number of terms,', len(Heff1))
#construct initial state (down down down)
spin_config = np.array([1,1,1])
psi1 =  sp_op.ini_state(ion_sys,s_state = spin_config, p_state = [[0,0,0]], state_type=1)
elist_com = [tensor(spin.sz(N,0),sp_op.p_I(ion_sys)),
          tensor(spin.sz(N,1),sp_op.p_I(ion_sys)),
          tensor(spin.sz(N,2),sp_op.p_I(ion_sys))]
#solve time dependent SE
times =  np.arange(0,40,1e-3)
print('______________________________________________________________________')
print('solving time-dependent Hamiltonian')
result1 = sesolve(Heff_com,psi1,times,args = arg_com,progress_bar=True,options=Options(nsteps=10000))      
#%%
#plot spin evolution
p1 = expect(elist_com[0],result1.states)
p2 = expect(elist_com[1],result1.states)
p3 = expect(elist_com[2],result1.states)
plt.plot(times,p1,label = 'site 1')
plt.plot(times,p2,label = 'site 2')
plt.plot(times,p3,label = 'site 3')
plt.xlabel(r'$t$ [ms]',fontsize = 14)
plt.ylabel(r'$<\sigma_{z}>$',fontsize = 14)
plt.ylim(-1,1)
plt.yticks(np.arange(-1,1.2,0.2),fontsize = 14)
plt.xticks(fontsize = 14)
plt.legend(fontsize = 12)
plt.grid()
plt.show()
#%%phonon evolution
#note to construct these phonon operators, any laser object above can be applied
mp_state1 = expect(sp_op.pstate_measure(ion_sys,laser1,5,0),result1.states) 
pplot = expect(sp_op.phonon_measure(ion_sys,laser1, mindex=0), result1.states)
print('Maximum phonon population of highest com phonon space')
print(np.max(mp_state1))
plt.plot(times,pplot,label = 'Phonon')
plt.xlabel(r'$t$ [ms]',fontsize = 14)
#title = r'$\delta_{com} = $'+str(ion_sys.delta)+' kHz'
plt.ylabel(r'$<a^+ a>$',fontsize = 14)
#plt.title(title,fontsize = 14)
plt.yticks(fontsize = 14)
plt.xticks(fontsize = 14)
plt.legend(fontsize = 12)
plt.grid()
plt.show()
#%%simulation with multi-modes, adjust numeric configuration of ion_sys
ion_sys.pcut = [[6,3,3]]
ion_sys.active_phonon = [[0,1,2]]
ion_sys.update_all()
ion_sys.list_para()
#%% Reproduce 5(a) construct Hamiltonian 
arg_all = Isp.H_td_argdic_general(ion0 = ion_sys, laser_list=[laser1,laser2,laser3,laser4])
Heff_all = Isp.H_td_multi_drives(ion0 = ion_sys, laser_list=[laser1,laser2,laser3,laser4],second_order=True,
                              rwa=True,arg_dic=arg_all,f_crit=rwa_fc ) #construct time-dependent H
#print('number of terms,', len(Heff1))
#construct initial state (down down down)
spin_config = np.array([1,1,1])
psi2 =  sp_op.ini_state(ion_sys,s_state = spin_config, p_state = [[0,0,0]], state_type=1)
elist_all = [tensor(spin.sz(N,0),sp_op.p_I(ion_sys)),
          tensor(spin.sz(N,1),sp_op.p_I(ion_sys)),
          tensor(spin.sz(N,2),sp_op.p_I(ion_sys))]
#solve time dependent SE
print('______________________________________________________________________')
print('solving time-dependent Hamiltonian')
result2 = sesolve(Heff_all,psi2,times,args = arg_all,progress_bar=True,options=Options(nsteps=10000))      
#%%
#plot spin evolution
p1 = expect(elist_all[0],result2.states)
p2 = expect(elist_all[1],result2.states)
p3 = expect(elist_all[2],result2.states)
plt.plot(times,p1,label = 'site 1')
plt.plot(times,p2,label = 'site 2')
plt.plot(times,p3,label = 'site 3')
plt.xlabel(r'$t$ [ms]',fontsize = 14)
plt.ylabel(r'$<\sigma_{z}>$',fontsize = 14)
plt.ylim(-1,1)
plt.yticks(np.arange(-1,1.2,0.2),fontsize = 14)
plt.xticks(fontsize = 14)
plt.legend(fontsize = 12)
plt.grid()
plt.show()
#%% Reproduce 6(a) 
#construct initial state (down up down)
spin_config = np.array([1,0,1])
psi3 =  sp_op.ini_state(ion_sys,s_state = spin_config, p_state = [[0,0,0]], state_type=1)
#solve time dependent SE
print('______________________________________________________________________')
print('solving time-dependent Hamiltonian')
result3 = sesolve(Heff_all,psi3,times,args = arg_all,progress_bar=True,options=Options(nsteps=10000))      
#%%
#plot spin evolution
p1 = expect(elist_all[0],result3.states)
p2 = expect(elist_all[1],result3.states)
p3 = expect(elist_all[2],result3.states)
plt.plot(times,p1,label = 'site 1')
plt.plot(times,p2,label = 'site 2')
plt.plot(times,p3,label = 'site 3')
plt.xlabel(r'$t$ [ms]',fontsize = 14)
plt.ylabel(r'$<\sigma_{z}>$',fontsize = 14)
plt.ylim(-1,1)
plt.yticks(np.arange(-1,1.2,0.2),fontsize = 14)
plt.xticks(fontsize = 14)
plt.legend(fontsize = 12)
plt.grid()
plt.show()
#%% Reproduce 6(d) 
#construct initial state (down down up)
spin_config = np.array([1,1,0])
psi4 =  sp_op.ini_state(ion_sys,s_state = spin_config, p_state = [[0,0,0]], state_type=1)
#solve time dependent SE
print('______________________________________________________________________')
print('solving time-dependent Hamiltonian')
result4 = sesolve(Heff_all,psi4,times,args = arg_all,progress_bar=True,options=Options(nsteps=10000))      
#%%
#plot spin evolution
p1 = expect(elist_all[0],result4.states)
p2 = expect(elist_all[1],result4.states)
p3 = expect(elist_all[2],result4.states)
plt.plot(times,p1,label = 'site 1')
plt.plot(times,p2,label = 'site 2')
plt.plot(times,p3,label = 'site 3')
plt.xlabel(r'$t$ [ms]',fontsize = 14)
plt.ylabel(r'$<\sigma_{z}>$',fontsize = 14)
plt.ylim(-1,1)
plt.yticks(np.arange(-1,1.2,0.2),fontsize = 14)
plt.xticks(fontsize = 14)
plt.legend(fontsize = 12)
plt.grid()
plt.show()