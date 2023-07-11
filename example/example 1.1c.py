# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:23:13 2022

@author: zhumj
Compute the full dynamics induced by a two-body ising coulping Hamiltonian with heating rate, 
compare the result with the dynamics under a pure spin interaction approximation.
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
from scipy import signal
import copy
import Qsim.ion_chain.interaction.dissipation as disp
#%%
'''
set parameters of the system, simulation for a phonon system with dissipation only, check if the Lindbladian is working
'''    
ion_sys = ions(trap_config={'N': 2, 'fx': 3, 'fz': 1}, 
                   numeric_config={'active_spin': [0],'active_phonon': [[0]], 'pcut': [[50]]},
                   )
ion_sys.list_para() #print parameters of the system
Bz = 0 #Effective magnetic field
N = ion_sys.N
#%%time evolution, assume the dissipation rate is 0.1 quanta/ms
clist = disp.heating(ion_sys,[0.1/(2*np.pi)],1) 
rho = sp_op.ini_state(ion_sys,[0],[[0]],0)
H0 = tensor(spin.zero_op(N=1),sp_op.p_zero(ion_sys))
times =  np.arange(0,1,10**(-4))
result0 = mesolve(H0,rho,times,clist,progress_bar=True,options=Options(nsteps=1000)) 
#%%plot phonon population 
df_p = 1 #for phonon measurements
sp_op.phonon_cutoff_error(result0.states, ion_sys, df=1, mindex=0,plot=True)
pplot = expect(sp_op.phonon_measure(ion_sys,df_p, mindex=0), result0.states)
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
#%%
'''
2body MS simulation for time-depedent Hamiltonian under RWA, drive the com mode,
assume a small heating rate of 0.1 quanta/ms  
'''
com_cut = 10; tilt_cut = 5
ion_sys.update_numeric(numeric_config={'active_spin': [0,1],'active_phonon': [[0,1]], 'pcut': [[com_cut,tilt_cut]]})
spin_config = np.array([0,0])
ion_sys.list_para()
#construct Hamiltonian 
laser1 = Laser(config = {'Omega_eff':30,'wavevector':1,'Dk':np.sqrt(2)*2*np.pi / (355*10**(-9)),
                         'laser_couple':[0,1], 'mu':50+1e3*ion_sys.fx,'phase':0})
laser1.list_para()
clist1 = disp.heating(ion_sys,[0.1/(2*np.pi),0],df=1) 
Heff,arg0 = iscc.H_ord(Bz,ion_sys,laser1) #construct time-dependent H
#construct initial state (initialized as up up)
rho1 = sp_op.ini_state(ion_sys,spin_config,[[0,0]],0)
elist1 = [tensor(spin.sz(N,0),sp_op.p_I(ion_sys)),tensor(spin.sz(N,1),sp_op.p_I(ion_sys)),
          sp_op.phonon_measure(ion_sys,df_p, mindex=0),
          sp_op.pstate_measure(ion_sys,df_p,com_cut-1,0)]
times =  np.arange(0,0.5,10**(-4))
#%%
#solve time dependent SE
print('______________________________________________________________________')
print('solving for complete Hamiltonian')
result1 = mesolve(Heff,rho1,times,clist1,args = arg0,progress_bar=True,options=Options(nsteps=1000)) 
#%% 
'''
simulation with a pure spin approximation
'''
psi0 = spin.spin_state(N,[0,0])  
J = iscp.Jt(ion_sys,laser1)
elist2 = [spin.sz(N,0),spin.sz(N,1)]
H = iscp.Hps(J,ion_sys,Bz)
print('______________________________________________________________________')
print('solving for pure spin interaction')
result = mesolve(H,psi0,times,e_ops=elist2,progress_bar=True, options=Options(nsteps=1000))
#%%
#plot spin dynamics
p0 = 0.5*(result.expect[0]+result.expect[1])
p1 = 0.5*(expect(elist1[0], result1.states)+expect(elist1[1], result1.states))
plt.plot(times,p0,label = 'Spin')
plt.plot(times,p1,label = 'Complete')
plt.xlabel(r'$t$ [ms]',fontsize = 14)
#title = r'$\delta_{com} = $'+str(ion_sys.delta)+' kHz'
plt.ylabel(r'$<\sigma_{zz}>$',fontsize = 14)
#plt.title(title,fontsize = 14)
plt.yticks(fontsize = 14)
plt.xticks(fontsize = 14)
plt.legend(fontsize = 12)
plt.grid()
plt.show()
#%%plot phonon population 
sp_op.phonon_cutoff_error(result1.states, ion_sys, df=1, mindex=0,plot=True)
pplot = expect(elist1[2], result1.states)
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
#%%simulate large heating rate, 3 quanta/ms consider com mode only
com_cut = 50
ion_sys.pcut= [[com_cut]]; ion_sys.active_phonon=[[0]]
ion_sys.list_para()
#construct Hamiltonian 
clist1 = disp.heating(ion_sys,[3/(2*np.pi),0],df=1) 
Heff,arg0 = iscc.H_ord(Bz,ion_sys,laser1) #construct time-dependent H
#construct initial state (initialized as up up)
rho1 = sp_op.ini_state(ion_sys,spin_config,[[0,0]],0)
elist2 = [tensor(spin.sz(N,0),sp_op.p_I(ion_sys)),tensor(spin.sz(N,1),sp_op.p_I(ion_sys)),
          sp_op.phonon_measure(ion_sys,df_p, mindex=0),
          sp_op.pstate_measure(ion_sys,df_p,com_cut-1,0)]
times =  np.arange(0,0.5,10**(-4))
#%%
#solve time dependent SE
print('______________________________________________________________________')
print('solving for complete Hamiltonian')
result2 = mesolve(Heff,rho1,times,clist1,args = arg0,progress_bar=True,options=Options(nsteps=1000)) 
#%%
#plot spin dynamics
p2 = 0.5*(expect(elist2[0], result2.states)+expect(elist2[1], result2.states))
plt.plot(times,p0,label = 'pure spin')
plt.plot(times,p2,label = 'heating')
plt.xlabel(r'$t$ [ms]',fontsize = 14)
#title = r'$\delta_{com} = $'+str(ion_sys.delta)+' kHz'
plt.ylabel(r'$<\sigma_{zz}>$',fontsize = 14)
#plt.title(title,fontsize = 14)
plt.yticks(fontsize = 14)
plt.xticks(fontsize = 14)
plt.legend(fontsize = 12)
plt.grid()
plt.show()
#%%plot phonon population 
sp_op.phonon_cutoff_error(result2.states, ion_sys, df=1, mindex=0,plot=True)
pplot2 = expect(elist2[2], result2.states)
plt.plot(times,pplot2,label = 'Phonon')
plt.xlabel(r'$t$ [ms]',fontsize = 14)
#title = r'$\delta_{com} = $'+str(ion_sys.delta)+' kHz'
plt.ylabel(r'$<a^+ a>$',fontsize = 14)
#plt.title(title,fontsize = 14)
plt.yticks(fontsize = 14)
plt.xticks(fontsize = 14)
plt.legend(fontsize = 12)
plt.grid()
plt.show()
