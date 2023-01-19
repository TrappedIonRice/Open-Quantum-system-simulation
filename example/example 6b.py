# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 14:14:49 2022
Compute ordinary frame phonon evolution for a anharmonic simulator of 3 ions, 
with laser coupled to ion 1 on the side in radial direction,  with anharmonic resonant coupling between
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
import copy
#%% set ion1 for drive 1, radial coupling to rock mode
ion1 = ions() #construct a two ion system using class ions 
ion1.N = 3
ion1.fx = 2
ion1.fz = (20/63)**0.5 * ion1.fx #set to resonant condition
ion1.df_laser = 1
ion1.delta_ref = 2
ion1.delta =-50
ion1.fr = 5*np.abs(ion1.delta);ion1.fb =  5*np.abs(ion1.delta)
ion1.laser_couple = [0]
ion1.pcut = [[100],[100]]
ion1.active_phonon = [[1],[2]] #only consider tilt for axial and rock for radial
ion1.list_para()
ion1.plot_freq()
#%%set up ion2, has same Hilbert space as ion1 but have laser couple to Axial tilt 
ion2 = copy.copy(ion1)
ion2.df_laser = 0
ion2.delta = np.sqrt(2)* ion1.delta
ion2.delta_ref = 1
#ion2.delta = ion1.delta
ion2.list_para()
ion2.plot_freq()
#%%
Omegax = np.abs(ion1.delta)
deltaE = ion1.delta
'''
time scale
'''
t_scale = np.abs(ion1.delta)
tplot = np.linspace(0,50,10000)
times = tplot*2*np.pi/t_scale
'''
measure axial tilt mode and radial rock mode population
'''
ket0 = exop.ini_state(ion1,[0],[[0],[0]],1)
op1 = exop.phonon_measure(ion1,0,0) #axial tilt up
op2 = exop.phonon_measure(ion1,0) #radial rock up 
elist = [exop.spin_measure(ion1,[0]),
      op1,op2,
      tensor(ket0*ket0.dag())]
'''
construct anharmonic coupling term
'''
print('coupling strenght [kHz]',ion1.ah_couple(2,2,1)*ion1.fz*1000)
ah_coef = ion1.ah_couple(2,2,1)*fr_conv(ion1.fz,'khz') #kHz
operator_a = exop.p_ladder(ion1,0,1,0) #creation operator on tilt axial mode
operator_b =  exop.p_ladder(ion1,0,0,1) #destory operator on rock radial mode
ah_oper =ah_coef*tensor(spin.sI(1),operator_a*operator_b*operator_b)

#%% simulation without anharmonic terms
Hce, arg0 = ah_t.H_ord2(Omegax,deltaE,ion1,ion2,False,0)
print("__________________________________________________________")
print("solving time evolution without anharmonic term")
result1 = sesolve(Hce,ket0,times,elist,args=arg0,progress_bar=True,options=Options(nsteps=100000))

spin_1 =  result1.expect[0] #spin population of initial state
ph_num1_1 = result1.expect[1] #axial tilt population
ph_num1_2 = result1.expect[2] #radial rock population
sigma_1 = result1.expect[3] #survival probablity  
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
p2 = fig1.add_subplot(122)
p2.plot(tplot,ph_num1_1,'-',label=r'Axial Tilt')
p2.plot(tplot,ph_num1_2,'-',label=r'Radial Rock')
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
Hce, arg0 = ah_t.H_ord2(Omegax,deltaE,ion1,ion2,True,ah_oper)
#Hce = ah_t.Htot(deltaE, ion1,1,False,False)
print("__________________________________________________________")
print("solving time evolution with anharmonic term")
result2 = sesolve(Hce,ket0,times,elist,args=arg0,progress_bar=True,options=Options(nsteps=100000))
#result2 = sesolve(Hce,ket0,times,elist,progress_bar=True,options=Options(nsteps=100000))
spin_2 =  result2.expect[0] #spin population of initial state
ph_num2_1 = result2.expect[1] #Axial Tilt population
ph_num2_2 = result2.expect[2] #Radial Rock population
sigma_2 = result2.expect[3] #dilution factor  
#%%plot result 
xend = 20
fig1 = plt.figure(figsize=(12,4))
p1 = fig1.add_subplot(121)
p1.plot(tplot,spin_2,'-')
title = 'Spin evolution with anharmonic terms'
p1.set_xlabel(r'$\delta t/(2\pi)$',fontsize = 14)
p1.set_ylabel(r'$P_{\uparrow}$',fontsize = 14)
p1.set_title(title)
p1.grid()   
plt.xlim(0,xend)
p1.tick_params(axis='x', labelsize=13)
p1.tick_params(axis='y', labelsize=13)
p2 = fig1.add_subplot(122)
p2.plot(tplot,ph_num2_1,'-',label=r'Axial Tilt')
p2.plot(tplot,ph_num2_2,'-',label=r'Radial Rock')
title = 'Phonon evolution with anharmonic terms'
p2.set_xlabel(r'$\delta t/(2\pi)$',fontsize = 14)
p2.set_ylabel(r'$<a^+a>$',fontsize = 14)
p2.set_title(title)
p2.grid()   
p2.tick_params(axis='x', labelsize=13)
p2.tick_params(axis='y', labelsize=13)
p2.legend()
plt.xlim(0,xend)
plt.show()
#%%
plt.clf()
plt.plot(tplot,spin_1,label = 'without anharmonicity')
plt.plot(tplot,spin_2,label = 'with anharmonicity')
plt.xlabel(r'$\delta t/(2\pi)$',fontsize = 14)
plt.ylabel(r'$P_{\uparrow}$',fontsize = 14)
plt.title('Spin evolution')
plt.grid()   
plt.ylim(0,1)
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
plt.ylim(0,1)  
plt.xlim(0,xend)
plt.yticks(fontsize = 13)
plt.xticks(fontsize = 13)
plt.legend()
plt.show()