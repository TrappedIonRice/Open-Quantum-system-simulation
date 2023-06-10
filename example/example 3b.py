# -*- coding: utf-8 -*-
"""
Compute the time evolution of a 3 ion system contructed to simulate excitation transfer 
between 2 sites in reasonant interaction frame and using time-dependent Hamiltonian in
ordinary interaction.
verify the results are the same.
@author: zhumj
"""
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import Qsim.operator.spin as spin
import Qsim.operator.phonon as phon
import Qsim.operator.spin_phonon as sp_op
import Qsim.ion_chain.transfer.exci_transfer as extrans
import Qsim.ion_chain.ising.ising_cex as iscex
from  Qsim.ion_chain.ion_system import *
#%%
'''
parameters of the system, in this example, we compute the evoluation at type 1
reasonance at Delta E = 1*delta_rock
'''       
ion_sys = ions() 
ion_sys.N = 3
ion_sys.df_laser = 1 #couple to Radial vibrational modes
ion_sys.laser_couple = [0,1] #laser applied to ion 1,2
ion_sys.coolant = [2] #ion 3 as coolant
ion_sys.delta_ref = 2
ion_sys.delta = -100
ion_sys.fr = 50; ion_sys.fb = 50
ion_sys.phase = np.pi/2
ion_sys.gamma = [0,0,10] #cool the rocking mode only
ion_sys.list_para() #print parameters
ion_sys.plot_freq()
#%%    
'''
parameters of the system, in this example, we compute the evoluation at type 1
reasonance at Delta E = 1*delta_rock
'''  
#ion_sys.pcut = [3,2,2]
#solve time evolution for a single energy splitting
J23 = 1
E1 = 100 #set energy difference as 100kHz (\delta_rock)
E2 = 0
V = 0
print('coupling strength between ion 1 and 2', J23, ' kHz *h')
print('site energy difference ', E1-E2, ' kHz *h')
tscale = J23      #use J as time scale
tplot0 = np.arange(0,1,0.01)
times0 =tplot0/tscale
#%%
print("__________________________________________________________")
print('simulating with H in resonant interaction frame')
ion_sys.pcut = [[2,3,6]]
ion_sys.active_phonon = [[0,1,2]] #consider com, tilt, and rock
ion_sys.check_phonon()
oplist = [sp_op.spin_measure(ion_sys,[0,1]),#spin population
          sp_op.phonon_measure(ion_sys,2)] # rock mode population
clist1 = sp_op.c_op(ion_sys,[0.01,0.01,0.01]) #collapse operator
rho0 = sp_op.rho_thermal(ion_sys,[[0.01,0.01,0.01]],False,[0,1])  #initial state
H1 = extrans.H_res(J23,(E1-E2)/2,0,V,ion_sys) #generate Hamiltonian
#result = mesolve(H0,rho0,times,clist1,[],progress_bar=True,options=Options(nsteps=10000))
result1 = mesolve(H1,rho0,times0,clist1,oplist,progress_bar=True,options=Options(nsteps=100000))
#%%
'''
Use complete Hamiltonian in ordinary interaction frame and only consider active 2 modes 
'''
ion_sys.pcut = [[3,6]]
ion_sys.active_phonon = [[1,2]] #consider tilt, and rock
ion_sys.check_phonon()
oplist = [sp_op.spin_measure(ion_sys,[0,1]),
          sp_op.phonon_measure(ion_sys,1)] #spin population
clist2 = sp_op.c_op(ion_sys,[0.01,0.01,0.01]) #collapse operator
rho0 = sp_op.rho_thermal(ion_sys,[[0.01,0.01,0.01]],False,[0,1])  #initial state
print("__________________________________________________________")
print('simulating with time-dependent H in ordinary interaction frame')
H2, arg0 = extrans.H_ord(J23,(E1-E2)/2,0,V,ion_sys) #generate Hamiltonian
#result = mesolve(H0,rho0,times,clist1,[],progress_bar=True,options=Options(nsteps=10000))
result2 = mesolve(H2,rho0,times0,clist2,oplist,args= arg0,progress_bar=True,options=Options(nsteps=100000))
#%% plot spin popluation
pplot1 =  result1.expect[0]
pplot2 =  result2.expect[0]
plt.figure(0)
plt.plot(tplot0,pplot1,'-',label='special frame')
plt.plot(tplot0,pplot2,'x',label=r'ordinary frame',markersize=2)   
plt.xlabel(r'$\omega_0t/(2\pi)$')
plt.ylabel(r'$p_{tot}$')
title = r'$\delta_{rock} = -100kHz, \Delta E = $'+str(E1) + r'$kHz , J=$'+str(J23)+r'$kHz$'
plt.title(title)
plt.yticks(fontsize = 13)
plt.xticks(fontsize = 13)
plt.legend(fontsize=12)
plt.grid()   
plt.show()

#%% plot spin popluation
pplot3 =  result1.expect[1]
pplot4 =  result2.expect[1]
plt.figure(0)
plt.plot(tplot0,pplot3,'-',label='special frame')
plt.plot(tplot0,pplot4,'-',label=r'ordinary frame',markersize=2)   
plt.xlabel(r'$t\delta_0$',fontsize = 14)
plt.ylabel(r'$<a^+a>$',fontsize = 14)
title = r'$\delta_{rock} = -100kHz, \Delta E = $'+str(E1) + r'$kHz , J=$'+str(J23)+r'$kHz$'
plt.title(title)
plt.yticks(fontsize = 13)
plt.xticks(fontsize = 13)
plt.legend(fontsize=12)
plt.grid()   
plt.show()