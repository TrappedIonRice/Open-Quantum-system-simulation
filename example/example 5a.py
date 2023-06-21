# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 14:14:49 2022
Compute quantum dynamics of anharmonic coupling based on a simulator of 3 ions, 
with a single radial laser drive coupled to all ions and resonant anharmonic coupling between
tilt axial mode and rocking radial mode. (m=n=3,p=2) 
Compare the results in ordinary and resonant frames.   

@author: zhumj
"""
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import Qsim.operator.spin as spin
import Qsim.operator.phonon as phon
import Qsim.operator.spin_phonon as sp_op
import Qsim.ion_chain.transfer.anharmonic_transfer as ah_t
from  Qsim.ion_chain.ion_system import *
#from to_precision import to_precision
import Qsim.ion_chain.transfer.chaos as chaos
import copy
from astropy.io import ascii
from astropy.table import Table
#%% set ion1 for drive 1, axial laser drive, this object is for constructing the 
#Hamiltonian in resonant frame.
delta0 = -80
cut_lev1 = 20; cut_lev2 = 40
ion1 = ions() 
ion1.N = 3
ion1.fx = 3.1
ion1.fz = (20/63)**0.5 * ion1.fx #resonant condition
ion1.df_laser = 0
ion1.delta_ref = 1
ion1.delta =  2*delta0 #this ensures the resonant condition holds in all frames
ion1.laser_couple = [0,1,2] #couple to all 3 ions
ion1.pcut = [[cut_lev1],[cut_lev2]]
ion1.active_phonon = [[1],[2]] #only consider tilt for axial and rock for radial
mconfig = [2,2,1]
f_fac = 0.5
ion1.fr = f_fac*np.abs(delta0); ion1.fb =  f_fac * np.abs(delta0) #rabi frequency
ion1.list_para()
ion1.plot_freq()
#%%set up ion2, it has same Hilbert space as ion1 but with radial laser drive
ion2 = copy.copy(ion1)
ion2.df_laser = 1
ion2.delta = delta0
ion2.delta_ref = 2
#ion2.delta = ion1.delta
ion2.list_para()
ion2.plot_freq()
#%%
deltaE0 = 2*ion2.delta
Omegax0 = 0.3*np.abs(ion2.delta)
Omegax = [Omegax0]*3
deltaE = [deltaE0]*3
'''
time scale
'''
t_scale = np.abs(ion2.delta)
tplot = np.linspace(0,20,5000)
times = tplot/t_scale
'''
operators for measuring 
'''
spin_config = np.array([0,1,0]) #initilaize spin state as up down up 
ket0 = sp_op.ini_state(ion1,spin_config,[[5],[5]],1) #phonon population n = 5
Lop = chaos.L_imbalance(spin_config,ion1) #spin imbalance 
p_op1 = sp_op.phonon_measure(ion2,0,0) #axial tilt mode population 
p_op2 = sp_op.phonon_measure(ion2,0) #radial rock mode population 
s_op1 = sp_op.site_spin_measure(ion2,0) #spin population of ion1,2,3 
s_op2 = sp_op.site_spin_measure(ion2,1)
s_op3 = sp_op.site_spin_measure(ion2,2)
'''
construct anharmonic coupling term
'''
print('coupling strenght [kHz]',ion2.ah_couple(mconfig)*ion2.fz*1000)
ah_coef = ion2.ah_couple(mconfig)*fr_conv(ion2.fz,'kHz') #kHz
operator_a = sp_op.p_ladder(ion2,0,1,0) #creation operator on tilt axial mode
operator_b =  sp_op.p_ladder(ion2,0,0,1) #destory operator on rock radial mode
ah_oper =ah_coef*tensor(spin.sI(ion2.df_spin),operator_a*operator_b*operator_b)
ndata = Table()
#%% dynamics in ordinary frame 
Hce, arg0 = ah_t.H_ord1(Omegax,deltaE,ion2,True,ah_oper)
print("__________________________________________________________")
print("solving time evolution in ordinary frame")
result1 = sesolve(Hce,ket0,times,args=arg0,progress_bar=True,options=Options(nsteps=100000))

#report error
mp_state1_1 = expect(sp_op.pstate_measure(ion2,cut_lev1-1,0,0),result1.states) 
mp_state2_1 = expect(sp_op.pstate_measure(ion2,cut_lev2-1,0),result1.states) 
print('Maximum phonon population of highest axial tilt space')
print(np.max(mp_state1_1))
print('Maximum phonon population of highest radial rock space')
print(np.max(mp_state2_1))
print('all simulation completed')  
#%%
fig1 = plt.figure(figsize=(12,4))
p1 = fig1.add_subplot(121)
p1.plot(tplot,expect(s_op1,result1.states),'x',label=r'ion 1')
p1.plot(tplot,expect(s_op2,result1.states),'-',label=r'ion 2')
p1.plot(tplot,expect(s_op3,result1.states),'--',label=r'ion 3')
p1.legend()
title = 'Spin evolution with anharmonic terms'
p1.set_xlabel(r'$\delta t/(2\pi)$',fontsize = 14)
p1.set_ylabel(r'$P_{\uparrow}$',fontsize = 14)
p1.set_title(title)
p1.grid()   
p1.tick_params(axis='x', labelsize=13)
p1.tick_params(axis='y', labelsize=13)
p2 = fig1.add_subplot(122)
p2.plot(tplot,expect(p_op1,result1.states),'-',label=r'Axial Tilt')
p2.plot(tplot,expect(p_op2,result1.states),'-',label=r'Radial Rock')
title = 'Phonon evolution with anharmonic terms'
p2.set_xlabel(r'$\delta t/(2\pi)$',fontsize = 14)
p2.set_ylabel(r'$<a^+a>$',fontsize = 14)
p2.set_title(title)
p2.grid()   
p2.tick_params(axis='x', labelsize=13)
p2.tick_params(axis='y', labelsize=13)
p2.legend()
plt.show()
#%%
Hres= ah_t.H_res1(Omegax,deltaE,ion2,ion1,True,ah_oper)
print("__________________________________________________________")
print("solving time evolution in resonant frame")
result2 = sesolve(Hres,ket0,times,progress_bar=True,options=Options(nsteps=100000))
#%%
mp_state1_2 = expect(sp_op.pstate_measure(ion2,cut_lev1-1,0,0),result2.states) 
mp_state2_2 = expect(sp_op.pstate_measure(ion2,cut_lev2-1,0),result2.states) 
print('Maximum phonon population of highest axial tilt space')
print(np.max(mp_state1_2))
print('Maximum phonon population of highest radial rock space')
print(np.max(mp_state2_2))
print('all simulation completed')  
#%% Compare the time evolution of spin imbalance in two frames
L1 = expect(Lop,result1.states) #spin imbalance 
L2 = expect(Lop,result2.states) 
plt.plot(tplot,L1 ,'x',label=r'ordinary')
plt.plot(tplot,L2 ,'-',label=r'resonant')
plt.xlabel(r'$\delta_0 t/(2\pi)$',fontsize = 14)
plt.ylabel(r'$\langle L(t) \rangle$',fontsize = 14)
plt.grid()   
plt.ylim(0,2)
plt.yticks(fontsize = 13)
plt.xticks(fontsize = 13)
plt.legend()
#plt.show()   
#plt.savefig('1drive L fsb.png',dpi=200)


