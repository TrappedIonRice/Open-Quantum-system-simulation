# -*- coding: utf-8 -*-
"""
Simulate two-modes schema for three-body-coupling in ordinary frame, consider com
mode only. 

@author: zhumj
"""

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
#%%
ion_sys = Ions_asy( trap_config = {'N':3,'fx':3,'fz':1,'offset':200},
                   numeric_config = {'active_spin':[0,1,2], 'active_phonon':[[0],[0]],'pcut' : [[10],[10]]},)

ion_sys.list_para()
#%%
delta = 5

 
laser_xr = Laser(config = {'Omega_eff':0,'wavevector':1,'Dk':np.sqrt(2)*2*np.pi / (355*10**(-9)),
                         'laser_couple':[0,1,2], 'mu':-1*(delta+1e3*ion_sys.fx),'phase':0})

Omega_eff = 10*laser_xr.eta(ion_sys.fx) 
laser_xr.Omega_eff =  Omega_eff

laser_xb = Laser(config = {'Omega_eff':Omega_eff,
                           'wavevector':1,'Dk':np.sqrt(2)*2*np.pi / (355*10**(-9)),
                         'laser_couple':[0,1,2], 
                         'mu':2*(delta+1e3*ion_sys.fx)
                         ,'phase':0})

laser_yr = Laser(config = {'Omega_eff':Omega_eff,
                           'wavevector':2,'Dk':np.sqrt(2)*2*np.pi / (355*10**(-9)),
                         'laser_couple':[0,1,2], 
                         'mu':-2*(delta+1e3*ion_sys.radial_freq2[0]),
                         'phase':0})

laser_yb = Laser(config = {'Omega_eff':Omega_eff,
                           'wavevector':2,'Dk':np.sqrt(2)*2*np.pi / (355*10**(-9)),
                         'laser_couple':[0,1,2], 
                         'mu': (delta+1e3*ion_sys.radial_freq2[0]),
                         'phase':0})

#we want the coupling coefficients to be equal to Omega_eff
laser_xb.Omega_eff = 2*np.sqrt(3)*Omega_eff/laser_xr.eta(ion_sys.fx) 
laser_yr.Omega_eff = 2*np.sqrt(3)*Omega_eff/laser_xr.eta(ion_sys.radial_freq2[0]) 
#print parameters
print('=====================================')
print('x Red Sideband')
laser_xr.list_para()
print('=====================================')
print('x Blue Sideband')
laser_xb.list_para()
print('=====================================')
print('y Red Sideband')
laser_yr.list_para()
print('=====================================')
print('y Blue Sideband')
laser_yb.list_para()
ion_sys.plot_freq(show_axial = False, show_neg =  True , 
                  laser_list=[laser_xr,laser_xb,laser_yr,laser_yb])
#%%Construct Hamiltonian
Heff,arg = iscc.H_com_asy(ion_sys, laser_xr, laser_xb, laser_yr, laser_yb)
N = ion_sys.N
spin_config = np.array([1,1,1])
psi1 = sp_op.ini_state(ion_sys,s_state = spin_config, p_state = [[0],[0]], state_type=1)
elist_com = [tensor(spin.sz(N,0),sp_op.p_I(ion_sys)),
          tensor(spin.sz(N,1),sp_op.p_I(ion_sys)),
          tensor(spin.sz(N,2),sp_op.p_I(ion_sys))]
#solve time dependent SE
times =  np.arange(0,200,1e-1)
print('______________________________________________________________________')
print('solving time-dependent Hamiltonian')
result1 = sesolve(Heff,psi1,times,args = arg,progress_bar=True,options=Options(nsteps=10000))   
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
#%% plot projection on GHZ state
s1 = spin.spin_state(N,[0,0,0])
s2 = spin.spin_state(N,[1,1,1])
splus = 1/np.sqrt(2)*(s1+1j*s2)
proj1=tensor(splus*splus.dag(),sp_op.p_I(ion_sys))
pghz = expect(proj1,result1.states)
plt.plot(times,pghz,label = 'no pa')
plt.xlabel(r'$t$ [ms]',fontsize = 14)
plt.ylabel(r'$p_+$',fontsize = 14)
#plt.ylim(-1,1)
#plt.yticks(np.arange(-1,1.2,0.2),fontsize = 14)
plt.xticks(fontsize = 14)
plt.legend(fontsize = 12)
plt.grid()
plt.show() 
#%%
sp_op.phonon_cutoff_error(result1.states, ion_sys, df=0, mindex=0,plot=True)
pplot = expect(sp_op.phonon_measure(ion_sys,0,mindex=0), result1.states)
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