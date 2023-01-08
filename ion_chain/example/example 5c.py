# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 14:14:49 2022
Compute ordinary frame phonon evolution for a anharmonic simulator of 3 ions, 
with laser coupled to ion 1 on the side in radial direction, resonant coupling between
tilt axial mode and rocking radial mode. Compute using sesolve
in this script, consider both tilt and rock mode for 2 vibrational directions

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
#%% set up the system using module for anharmonic transfer using 3 ions
'''
Set up the system 
'''    

ion_sys = ions() #construct a two ion system using class ions 
ion_sys.N = 3
ion_sys.fx = 2
ion_sys.fz = (20/63)**0.5 * ion_sys.fx #set to resonant condition
ion_sys.delta_ref = 2
ion_sys.delta =-200*np.sqrt(2)
Omegax = 0.25*np.abs(ion_sys.delta)
ion_sys.fr = 0.8*np.abs(ion_sys.delta); ion_sys.fb = 0.8*np.abs(ion_sys.delta)
ion_sys.laser_couple = [0]
ion_sys.pcut = [[10,8],[8,10]]
ion_sys.active_phonon = [[1,2],[1,2]] #only consider tilt, rock mode
deltaE = np.sqrt(3)*ion_sys.delta
'''
time scale
'''
t_scale = np.abs(ion_sys.delta)
tplot = np.linspace(0,1,1000)
times = tplot*2*np.pi/t_scale
'''
measure axial tilt mode and radial rock mode population
'''
ket0 = exop.ini_state(ion_sys,1,1)
op1 = exop.phonon_measure(ion_sys,0,0) #axial tilt up
op2 = exop.phonon_measure(ion_sys,1,0) #axial rock up
op3 = exop.phonon_measure(ion_sys,0) #radial tilt up
op4 = exop.phonon_measure(ion_sys,1) #radial rock up 
elist = [exop.spin_measure(ion_sys,[0]),
        op1,op2,op3,op4,
        tensor(ket0*ket0.dag())]
ion_sys.list_para()
ion_sys.plot_freq()
'''
construct anharmonic coupling term
'''
print('coupling strenght [kHz]',ion_sys.ah_couple(2,2,1)*ion_sys.fz*1000)
ah_coef = ion_sys.ah_couple(2,2,1)*fr_conv(ion_sys.fz,'khz') #kHz
operator_a = exop.p_ladder(ion_sys,0,1,0) #creation operator on tilt axial mode
operator_b =  exop.p_ladder(ion_sys,1,0,1) #destory operator on rock radial mode
ah_oper =ah_coef*tensor(spin.sI(1),operator_a*operator_b*operator_b)
#%% simulation without anharmonic terms
Hce, arg0 = ah_t.H_ord(Omegax,deltaE, ion_sys,True,False,0)
#Hce = ah_t.Htot(deltaE, ion_sys,1,False,False)
print("__________________________________________________________")
print("solving time evolution without anharmonic term")
result1 = sesolve(Hce,ket0,times,elist,args=arg0,progress_bar=True,options=Options(nsteps=100000))
#result2 = sesolve(Hce,ket0,times,elist,progress_bar=True,options=Options(nsteps=100000))
spin_1 =  result1.expect[0] #spin population of initial state
ph_num1_1 = result1.expect[1] #axial tilt population
ph_num1_2 = result1.expect[2] #axial rock population
ph_num1_3 = result1.expect[3] #radial tilt population
ph_num1_4 = result1.expect[4] #radial rock population
sigma_1 = result1.expect[5] #dilution factor  
#%%plot result    
fig1 = plt.figure(figsize=(12,4))
p1 = fig1.add_subplot(121)
p1.plot(tplot,spin_1,'-')
title = 'Spin evolution without anharmonic terms'
p1.set_xlabel(r'$\delta t/(2\pi)$',fontsize = 14)
p1.set_ylabel(r'$P_{\uparrow}$',fontsize = 14)
p1.set_title(title)
p1.grid()   
#plt.xlim(0,20)
p1.tick_params(axis='x', labelsize=13)
p1.tick_params(axis='y', labelsize=13)
p1.legend()
p2 = fig1.add_subplot(122)
p2.plot(tplot,ph_num1_1,'-',label=r'Axial Tilt')
p2.plot(tplot,ph_num1_2,'-',label=r'Axial Rock')
p2.plot(tplot,ph_num1_3,'-',label=r'Radial Tilt')
p2.plot(tplot,ph_num1_4,'-',label=r'Radial Rock')
title = 'Phonon evolution without anharmonic terms'
p2.set_xlabel(r'$\delta t/(2\pi)$',fontsize = 14)
p2.set_ylabel(r'$<a^+a>$',fontsize = 14)
p2.set_title(title)
p2.grid()   
p2.tick_params(axis='x', labelsize=13)
p2.tick_params(axis='y', labelsize=13)
p2.legend()
plt.show()
#%%
Hce, arg0 = ah_t.H_ord(Omegax, deltaE, ion_sys,True,True,ah_oper)
#Hce = ah_t.Htot(deltaE, ion_sys,1,False,False)
print("__________________________________________________________")
print("solving time evolution with anharmonic term")
result2 = sesolve(Hce,ket0,times,elist,args=arg0,progress_bar=True,options=Options(nsteps=100000))
#result2 = sesolve(Hce,ket0,times,elist,progress_bar=True,options=Options(nsteps=100000))
spin_2 =  result2.expect[0] #spin population of initial state
ph_num2_1 = result2.expect[1] #axial tilt population
ph_num2_2 = result2.expect[2] #axial rock population
ph_num2_3 = result2.expect[3] #radial tilt population
ph_num2_4 = result2.expect[4] #radial rock population
sigma_2 = result2.expect[5] #dilution factor  
#%%plot result    
fig1 = plt.figure(figsize=(12,4))
p1 = fig1.add_subplot(121)
p1.plot(tplot,spin_2,'-')
title = 'Spin evolution with anharmonic terms'
p1.set_xlabel(r'$\delta t/(2\pi)$',fontsize = 14)
p1.set_ylabel(r'$P_{\uparrow}$',fontsize = 14)
p1.set_title(title)
p1.grid()   
#plt.xlim(0,20)
p1.tick_params(axis='x', labelsize=13)
p1.tick_params(axis='y', labelsize=13)
p1.legend()
p2 = fig1.add_subplot(122)
p2.plot(tplot,ph_num2_1,'-',label=r'Axial Tilt')
p2.plot(tplot,ph_num2_2,'-',label=r'Axial Rock')
p2.plot(tplot,ph_num2_3,'-',label=r'Radial Tilt')
p2.plot(tplot,ph_num2_4,'-',label=r'Radial Rock')
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
plt.clf()
plt.plot(tplot,spin_1,label = 'without anharmonicity')
plt.plot(tplot,spin_2,label = 'with anharmonicity')
plt.xlabel(r'$\delta t/(2\pi)$',fontsize = 14)
plt.ylabel(r'$P_{\uparrow}$',fontsize = 14)
plt.title('Spin evolution')
plt.grid()   
#plt.xlim(0,20)
plt.yticks(fontsize = 13)
plt.xticks(fontsize = 13)
plt.legend()
plt.show()
#%%
plt.clf()
plt.plot(tplot,sigma_1,label = 'without anharmonicity')
plt.plot(tplot,sigma_2,'.',label = 'with anharmonicity')
plt.xlabel(r'$\delta t/(2\pi)$',fontsize = 14)
plt.ylabel(r'$P_{\uparrow 0}$',fontsize = 14)
plt.title(r'survival probability')
plt.grid()   
#plt.xlim(0,20)
plt.yticks(fontsize = 13)
plt.xticks(fontsize = 13)
plt.legend()
plt.show()