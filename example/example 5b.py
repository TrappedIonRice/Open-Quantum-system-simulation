# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 14:14:49 2022
Compute quantum dynamics of anharmonic coupling based on a simulator of 3 ions, 
with a single radial laser drive coupled to all ions and off-resonant anharmonic coupling between
tilt axial mode and rocking radial mode. (m=n=3,p=2) 
Compare the dynamics by two slightly different H   

@author: zhumj
"""
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import Qsim.operator.spin as spin
import Qsim.operator.phonon as phon
import Qsim.operator.spin_phonon as sp_op
import Qsim.ion_chain.transfer.anharmonic_transfer as ah_t
import Qsim.ion_chain.transfer.chaos as chaos
from  Qsim.ion_chain.ion_system import *
#from to_precision import to_precision
import copy
from astropy.io import ascii
from astropy.table import Table

#%% ion1 for axial laser coupling
ion1 = ions() 
ion1.N = 3
ion1.fx = 3.1
delta0 = 95 #scale for harmonic detuning
delta_ah0 = 80 #scale for anharmonic detuning 
d_fac = -5 #
ion1.fz = ah_t.res_freq(ion1.fx,d_fac*delta_ah0 ) #set to off-resonant condition (400kHz)
cut_lev1 = 25; cut_lev2 = 45
ion1.df_laser = 0
ion1.delta_ref = 1
fsb_fac1  = 0.5
ion1.laser_couple = [0,1,2] #couple to all 3 ions
ion1.pcut = [[cut_lev1],[cut_lev2]]
ion1.active_phonon = [[1],[2]] #only consider tilt for axial and rock for radial
tplot = np.linspace(0,20,5000)
spin_config = np.array([0,1,0]) #initilaize spin state as up down up 
ket0 = sp_op.ini_state(ion1,spin_config,[[5],[5]],1) #phonon population n = 5
Lop = chaos.L_imbalance(spin_config,ion1) #
elist = [Lop]
mconfig = [2,2,1]
Delta0 = 2*delta0 - d_fac*delta_ah0  #compute the harmonic coupling coefficeint Delta_0 in res frame
print('harmonic coupling coefficeint Delta_0: ', Delta0)
ion1.delta =  -1 * Delta0 
ion1.fr = fsb_fac1*np.abs(delta0); ion1.fb =  fsb_fac1 *np.abs(delta0)
ion1.list_para()
#%%ion2 for radial laser coupling
ion2 = copy.copy(ion1)
ion2.df_laser = 1
ion2.delta = -1* delta0
ion2.delta_ref = 2
ion2.list_para()
#%% time scale and other parameters in H
t_scale =  delta_ah0
times = tplot/t_scale
deltaE0 = 2*ion2.delta
Omegax0 = 0.3*np.abs(ion2.delta)
Omegax = [Omegax0]*3
deltaE = [deltaE0]*3
'''
construct anharmonic coupling term
'''
ah_coef = ion2.ah_couple(mconfig)*fr_conv(ion2.fz,'kHz') #kHz
operator_a = sp_op.p_ladder(ion2,0,0,0) #destory operator on tilt axial mode
operator_b =  sp_op.p_ladder(ion2,0,1,1) #create operator on rock radial mode
#term 1, a * b^+ * b
ah_oper1 =ah_coef*2*tensor(spin.sI(ion2.df_spin),operator_a*operator_b*operator_b.dag())
#term 2, a * b^+ * b^+
ah_oper2 =ah_coef*tensor(spin.sI(ion2.df_spin),operator_a*operator_b*operator_b)
#%% dynamics in resonant frame 
print("solving time evolution in resonant frame")
Hres = ah_t.H_res1(Omegax,deltaE,ion2,ion1,True,ah_oper2)
result1 = sesolve(Hres,ket0,times,progress_bar=True,options=Options(nsteps=100000))
L_1 = expect(elist[0],result1.states) #spin imbalance  
#%% dynamics with time-depedent anharmonic term in ordinary frame 
print("solving time evolution with time-depedent anharmonic term")
Hce1, arg1 = ah_t.H_ord1(Omegax,deltaE,ion2,True,[ah_oper1,ah_oper2],True)
print('anharmonic term frequency 1: '+str(np.round(np.abs(arg1['d1'])/(2*np.pi),2))+ ' kHz')
print('anharmonic term frequency 2: '+str(np.round(np.abs(arg1['d2'])/(2*np.pi),2))+ ' kHz')
result2_1 = sesolve(Hce2,ket0,times,args=arg1,progress_bar=True,options=Options(nsteps=100000))
L2_1 = expect(elist[0],result2_1.states) #spin imbalance   
#%%compare dynamics in two frames
plt.plot(tplot,L2_1 ,'-',label=r'ordinary frame')
plt.plot(tplot,L_1 ,'-',label=r'resonant frame')
plt.xlabel(r'$\delta_0 t/(2\pi)$',fontsize = 14)
plt.ylabel(r'$\langle L(t) \rangle$',fontsize = 14)
plt.grid()   
plt.yticks(fontsize = 13)
plt.xticks(fontsize = 13)
plt.legend()
plt.show()
#%%slightly change H and compute dynamics
ion1.fr = ion1.fr*(1+1e-2); ion1.fb =  ion1.fb*(1+1e-2)
ion2.fr = ion2.fr*(1+1e-2); ion2.fb =  ion2.fb*(1+1e-2)
print("solving time evolution in ordinary frame with slightly different H")
Hce2, arg2 =ah_t.H_ord1(Omegax,deltaE,ion2,True,[ah_oper1,ah_oper2],True)
result2_2 = sesolve(Hce2,ket0,times,args=arg2,progress_bar=True,options=Options(nsteps=100000))
L2_2 = expect(elist[0],result2_2.states) #spin imbalance    
dL = L2_1 - L2_2 #difference between spin imbalance
p_ovr = np.array([]) #over lap between the same initial state
for i in range(len(tplot)):
    p_ovr = np.append(p_ovr,(np.abs(result2_1.states[i].overlap(result2_2.states[i])))**2)
#%%plot result
fig1 = plt.figure(figsize=(12,4))
p1 = fig1.add_subplot(121)
p1.plot(tplot,dL,'-',label=r'difference in spin imbalance')
p1.legend()
p1.set_xlabel(r'$\delta_0 t/(2\pi)$',fontsize = 14)
p1.set_ylabel(r'$\Delta L_{\psi}(t)$',fontsize = 14)
p1.grid()   
p1.tick_params(axis='x', labelsize=13)
p1.tick_params(axis='y', labelsize=13)
p2 = fig1.add_subplot(122)
p2.plot(tplot,p_ovr,'-',label=r'state overlap')
p2.set_xlabel(r'$\delta_0 t/(2\pi)$',fontsize = 14)
p2.set_ylabel(r'$\Delta P_{\psi}(t)$',fontsize = 14)
p2.grid()   
p2.tick_params(axis='x', labelsize=13)
p2.tick_params(axis='y', labelsize=13)
p2.legend()
plt.show() 