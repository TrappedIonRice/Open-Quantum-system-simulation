# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 14:14:49 2022
Compute ordinary/resonant frame phonon evolution for a anharmonic simulator of 3 ions, 
with laser coupled to ion 1 on the side in radial direction,  with anharmonic resonant coupling between
tilt axial mode and rocking radial mode. 
in this script, consider both tilt and rock mode for 2 vibrational directions
Check the consistency between the two frames
@author: zhumj
"""
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import Qsim.operator.spin as spin
import Qsim.operator.phonon as phon
import Qsim.ion_chain.transfer.exci_operators as exop
import Qsim.ion_chain.transfer.anharmonic_transfer as ah_t
from  Qsim.ion_chain.ion_system import *
import copy
#%% set ion1 for drive 1, Axial coupling to tilt mode
delta0 = -50
ion1 = ions() #construct a two ion system using class ions 
ion1.N = 3
ion1.fx = 2
ion1.fz = (20/63)**0.5 * ion1.fx #set to resonant condition
ion1.df_laser = 0
ion1.delta_ref = 1
ion1.delta =np.sqrt(2)*delta0
ion1.fr = 5*np.abs(delta0);ion1.fb =  5*np.abs(delta0)
ion1.laser_couple = [0]
ion1.pcut = [[80],[80]]
ion1.active_phonon = [[1],[2]] #only consider tilt for axial and rock for radial
ion1.list_para()
ion1.plot_freq()
#%%set up ion2, has same Hilbert space as ion1 but have laser couple to radial rock
ion2 = copy.copy(ion1)
ion2.df_laser = 1
ion2.delta = delta0
ion2.delta_ref = 2
#ion2.delta = ion1.delta
ion2.list_para()
ion2.plot_freq()
#%%
Omegax = np.abs(ion2.delta)
deltaE = ion2.delta
'''
time scale
'''
t_scale = np.abs(ion2.delta)
tplot = np.linspace(0,5,5000)
times = tplot*2*np.pi/t_scale
'''
measure axial tilt mode and radial rock mode population
'''
ket0 = exop.ini_state(ion2,[0],[[0],[0]],1)
op1 = exop.phonon_measure(ion2,0,0) #axial tilt up
op2 = exop.phonon_measure(ion2,0) #radial rock up 
elist = [exop.spin_measure(ion2,[0]),
      op1,op2,
      tensor(ket0*ket0.dag())]
'''
construct anharmonic coupling term
'''
print('coupling strenght [kHz]',ion2.ah_couple(2,2,1)*ion2.fz*1000)
ah_coef = ion2.ah_couple(2,2,1)*fr_conv(ion2.fz,'khz') #kHz
operator_a = exop.p_ladder(ion2,0,1,0) #creation operator on tilt axial mode
operator_b =  exop.p_ladder(ion2,0,0,1) #destory operator on rock radial mode
ah_oper =ah_coef*tensor(spin.sI(1),operator_a*operator_b*operator_b)

#%%
''' 
simulation without anharmonic terms
'''
#ordinary frame
Hce, arg0 = ah_t.H_ord2(Omegax,deltaE,ion1,ion2,False,0)
#%%
print("__________________________________________________________")
print("solving time evolution without anharmonic term in ordinary frame")
result11 = sesolve(Hce,ket0,times,elist,args=arg0,progress_bar=True,options=Options(nsteps=100000))

spin_11 =  result11.expect[0] #spin population of initial state
ph_num1_11 = result11.expect[1] #axial tilt population
ph_num1_21 = result11.expect[2] #radial rock population
sigma_11 = result11.expect[3] #survival probablity  
#%% resonant frame
Hres= ah_t.H_res2(Omegax,deltaE,ion1,ion2,False,0)
print("__________________________________________________________")
print("solving time evolution without anharmonic term in resonant frame")
result12 = sesolve(Hres,ket0,times,elist,progress_bar=True,options=Options(nsteps=100000))

spin_12 =  result12.expect[0] #spin population of initial state
ph_num1_12 = result12.expect[1] #axial tilt population
ph_num1_22 = result12.expect[2] #radial rock population
sigma_12 = result12.expect[3] #survival probablity  
#%%plot result    
fig1 = plt.figure(figsize=(12,4))
p1 = fig1.add_subplot(121)
p1.plot(tplot,spin_11,'x',label='ordinary')
p1.plot(tplot,spin_12,'-',label='resonant')
p1.legend()
title = 'Spin evolution without anharmonic terms'
p1.set_xlabel(r'$\delta t/(2\pi)$',fontsize = 14)
p1.set_ylabel(r'$P_{\uparrow}$',fontsize = 14)
p1.set_title(title)
p1.grid()   
#plt.xlim(0,1)
p1.tick_params(axis='x', labelsize=13)
p1.tick_params(axis='y', labelsize=13)
p2 = fig1.add_subplot(122)
p2.plot(tplot,ph_num1_11,'x',label=r'Axial Tilt,ord')
p2.plot(tplot,ph_num1_21,'x',label=r'Radial Rock,ord')
p2.plot(tplot,ph_num1_12,'-',label=r'Axial Tilt,res')
p2.plot(tplot,ph_num1_22,'-',label=r'Radial Rock,res')
title = 'Phonon evolution without anharmonic terms'
p2.set_xlabel(r'$\delta t/(2\pi)$',fontsize = 14)
p2.set_ylabel(r'$<a^+a>$',fontsize = 14)
p2.set_title(title)
p2.grid()   
p2.tick_params(axis='x', labelsize=13)
p2.tick_params(axis='y', labelsize=13)
p2.legend()
plt.show()