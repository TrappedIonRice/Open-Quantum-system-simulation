# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:23:13 2022

@author: zhumj
Compute the dynamics induced by 3 body coupling Hamiltonian. Consider modes in 
both x y radial directions 
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
#%% set parameters of ion chain, simulation with com mode only
ion_sys = Ions_asy(trap_config={'N': 3, 'fx': 5, 'fz': 1,'offset':200}, 
                   numeric_config={'active_spin': [0, 1, 2],'active_phonon': [[0],[0]], 'pcut': [[6],[6]]},
                   )
ion_sys.list_para() #print parameters of the system
#%% set parameters of lasers in x
R1 = 26*1e3*2*np.pi #recoil frequency
#compute Dk
Dk1 = np.sqrt(R1*2*MYb171/h) #effective wavenumber
q = 1.3 #compensation scale parameter
delta = 2 #kHz
scale_x = 1 #set this to 0 to disable lasers in x direction
#symmetric beatnote
#blue sideband 1
laserx1 = Laser(config = {'Omega_eff':30,'wavevector':1,'Dk':Dk1,'laser_couple':[0,1,2],
                'mu':2*(delta+1e3*ion_sys.fx),'phase':0})
Omega_r = 5*laserx1.eta(ion_sys.fx)
Omega_b = 2*np.sqrt(3)*Omega_r/laserx1.eta(ion_sys.fx)
laserx1.Omega_eff = Omega_b * scale_x 
#red sideband
laserx2 = copy.copy(laserx1)
laserx2.Omega_eff = Omega_r * scale_x 
laserx2.mu = -1*(delta+1e3*ion_sys.fx)
#compensation beatnote
#compensation blue sideband
laserx3 = copy.copy(laserx1)
laserx3.Omega_eff = np.sqrt(q)*Omega_b*scale_x  
laserx3.mu = 2*(-q*delta+1e3*ion_sys.fx)
#compensation red sideband
laserx4 = copy.copy(laserx1)
laserx4.Omega_eff = np.sqrt(q)*Omega_r*scale_x  
laserx4.mu = -1*(-q*delta+1e3*ion_sys.fx)
print('________________________________________________')
print('Parameteres of laser 1')
laserx1.list_para()
print('________________________________________________')
print('Parameteres of laser 2')
laserx2.list_para()
print('________________________________________________')
print('Parameteres of laser 3')
laserx3.list_para()
print('________________________________________________')
print('Parameteres of laser 4')
laserx4.list_para()
#print(laser1.R/(2*np.pi))
print(laserx2.Omega(ion_sys)/(2*np.pi))
#%% set parameters of lasers in y
R2 = 26*1e3*2*np.pi #recoil frequency
#compute Dk
Dk2 = np.sqrt(R2*2*MYb171/h) #effective wavenumber
q = 1.3 #compensation scale parameter
delta = 2 #kHz
scale_y = 1 #set this to 0 to disable lasers in y direction
#symmetric beatnote
#blue sideband 1
lasery1 = Laser(config = {'Omega_eff':30,'wavevector':2,'Dk':Dk2,'laser_couple':[0,1,2],
                'mu':2*(delta+1e3*ion_sys.radial_freq2[0]),'phase':0})
Omega_b = 2*np.sqrt(3)*Omega_r/lasery1.eta(ion_sys.radial_freq2[0])
lasery1.Omega_eff = Omega_b*scale_y 
#red sideband
lasery2 = copy.copy(lasery1)
lasery2.Omega_eff = Omega_r*scale_y
lasery2.mu = -1*(delta+1e3*ion_sys.radial_freq2[0])
#compensation beatnote
#compensation blue sideband
lasery3 = copy.copy(lasery1) 
lasery3.Omega_eff = np.sqrt(q)*Omega_b *scale_y
lasery3.mu = 2*(-q*delta+1e3*ion_sys.radial_freq2[0])
#compensation red sideband
lasery4 = copy.copy(lasery1)
lasery4.Omega_eff = np.sqrt(q)*Omega_r*scale_y
lasery4.mu = -1*(-q*delta+1e3*ion_sys.radial_freq2[0])
print('________________________________________________')
print('Parameteres of laser 1')
lasery1.list_para()
print('________________________________________________')
print('Parameteres of laser 2')
lasery2.list_para()
print('________________________________________________')
print('Parameteres of laser 3')
lasery3.list_para()
print('________________________________________________')
print('Parameteres of laser 4')
lasery4.list_para()
#print(laser1.R/(2*np.pi))
print(lasery2.Omega(ion_sys)/(2*np.pi))
N = 3
#%%compute rwa critirion
delta_omega = (np.max(ion_sys.radial_freq)- np.min(ion_sys.radial_freq))*1000
rwa_fc = fr_conv(2*q*(delta_omega+delta+10),'Hz') 
print('RWA critrion frequency: '+ str(np.round(rwa_fc/(2*np.pi),2))+' [kHz]')
#10 is an additional factor to make sure all frequencies are included
#%% construct Hamiltonian 
laser_list_x = [laserx1,laserx2,laserx3,laserx4]
laser_list_y = [lasery1,lasery2,lasery3,lasery4]
arg_com_x = Isp.H_td_argdic_general(ion0 = ion_sys, laser_list=laser_list_x)
arg_com_y = Isp.H_td_argdic_general(ion0 = ion_sys, laser_list=laser_list_y)
arg_com = arg_com_x | arg_com_y
Heff_com = Isp.H_td_multi_drives_asy(ion_sys, laser_list=[laser_list_x,laser_list_y],
                                     second_order=True,
                              rwa=True,arg_dic=arg_com,f_crit=rwa_fc ) #construct time-dependent H
#print('number of terms,', len(Heff1))
#construct initial state (down down down)
spin_config = np.array([1,1,1])
psi1 =  sp_op.ini_state(ion_sys,s_state = spin_config, p_state = [[0],[0]], state_type=1)
elist_com = [tensor(spin.sz(N,0),sp_op.p_I(ion_sys)),
          tensor(spin.sz(N,1),sp_op.p_I(ion_sys)),
          tensor(spin.sz(N,2),sp_op.p_I(ion_sys))]
#%%solve time dependent SE
times =  np.arange(0,40,1e-3)
print('______________________________________________________________________')
print('solving SE with time-dependent H')
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
#x phonon modes
sp_op.phonon_cutoff_error(result1.states, ion_sys, df=1, mindex=0,plot=True)
pplot = expect(sp_op.phonon_measure(ion_sys,1,mindex=0), result1.states)
plt.plot(times,pplot,label = 'Phonon')
plt.xlabel(r'$t$ [ms]',fontsize = 14)
#title = r'$\delta_{com} = $'+str(ion_sys.delta)+' kHz'
plt.ylabel(r'$<a^+ a>$(x)',fontsize = 14)
#plt.title(title,fontsize = 14)
plt.yticks(fontsize = 14)
plt.xticks(fontsize = 14)
plt.legend(fontsize = 12)
plt.grid()
plt.show()
#%%phonon evolution
#y phonon modes
sp_op.phonon_cutoff_error(result1.states, ion_sys, df=2, mindex=0,plot=True)
pplot = expect(sp_op.phonon_measure(ion_sys,2,mindex=0), result1.states)
plt.plot(times,pplot,label = 'Phonon')
plt.xlabel(r'$t$ [ms]',fontsize = 14)
#title = r'$\delta_{com} = $'+str(ion_sys.delta)+' kHz'
plt.ylabel(r'$<a^+ a>$(y)',fontsize = 14)
#plt.title(title,fontsize = 14)
plt.yticks(fontsize = 14)
plt.xticks(fontsize = 14)
plt.legend(fontsize = 12)
plt.grid()
plt.show()