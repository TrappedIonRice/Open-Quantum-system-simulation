# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:23:13 2022

@author: zhumj
Compute the time evolution of the ising coulping with a complete Hamiltonian and
compare the result under a pure spin interaction approximation
"""
#%%
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import ion_chain.operator.spin as spin
import ion_chain.operator.phonon as phon
import ion_chain.ising.ising_ps as iscp
import ion_chain.ising.ising_c as iscc
from  ion_chain.ising.ion_system import *
from scipy import signal
#%%
'''
set parameters of the system
'''    
ion_sys = ions()
N = 2
ion_sys.N = 2 # 2 ion system 
ion_sys.fz = 0.2; #axial COM (Confining) frequency MHz
ion_sys.fx= 5; #transverse COM (Confining) frequency MHz
ion_sys.fr  = 30 #side band rabi frequency kHz
ion_sys.fb  = 30
pcut = [3,3]
ion_sys.pcut = pcut #truncation of phonon 
#delta = float(input("Enter detuning frequency (kHz): "))
ion_sys.delta_ref = 1 
delta = 100 #detuning from com mode
ion_sys.delta = delta
ion_sys.list_para() #print parameters of the system
Bz = 0 #Effective magnetic field
#%%
'''
simulation for complete Hamiltonian
'''
#construct Hamiltonian 
#time indepenedent part
H0 = tensor(iscp.HBz(N,Bz),phon.pI(pcut,N))
Heff,arg0 = iscc.Htot(H0,ion_sys) #construct time-dependent H
#construct initial state
nlist0 = [0,0,0]
psi1 = tensor(spin.phid(N),phon.phip(pcut,N,nlist0))
elist1 = [tensor(spin.sz(N,0),phon.pI(pcut,N)),tensor(spin.sz(N,1),phon.pI(pcut,N))]
#solve time dependent SE
times =  np.arange(0,4,10**(-4))
print('______________________________________________________________________')
print('solving for complete Hamiltonian')
result1 = mesolve(Heff,psi1,times,e_ops=elist1,args = arg0,progress_bar=True,options=Options(nsteps=1000))    
#%% 
'''
simulation with a pure spin approximation
'''
psi0 = spin.phid(N)  
J = iscp.Jt(ion_sys)
elist2 = [spin.sz(N,0),spin.sz(N,1)]
H = iscp.Hps(J,N,Bz)
print('______________________________________________________________________')
print('solving for pure spin interaction')
result = mesolve(H,psi0,times,e_ops=elist2,progress_bar=True, options=Options(nsteps=1000))
#%%
#plot result
p1 = 0.5*(result.expect[0]+result.expect[1])
p2 = 0.5*(result1.expect[0]+result1.expect[1])
plt.plot(times,p1,label = 'Spin')
plt.plot(times,p2,label = 'Complete')
plt.xlabel(r'$t$ [ms]',fontsize = 14)
title = r'$\delta_{com} = $'+str(delta)+' kHz'
plt.ylabel(r'$<\sigma_{zz}>$',fontsize = 14)
plt.title(title,fontsize = 14)
plt.yticks(fontsize = 14)
plt.xticks(fontsize = 14)
plt.legend(fontsize = 12)
plt.grid()
plt.show()
#%% extract frequency of oscillation for the complete evolution
dlist = ion_sys.dmlist()/(2*np.pi)
f, Pxx_den = signal.periodogram(p1-p2, 10000)
plt.semilogy(f, Pxx_den,)
label1 = r'$\delta_{com}=$' + str(dlist[0]) + ' kHz'
label2 = '$\delta_{1}=$' + str(np.round(dlist[1])) + ' kHz'
plt.plot([dlist[0],dlist[0]],[0,1],label=label1)
plt.plot([dlist[1],dlist[1]],[0,1],label=label2)
#plt.ylim([1e-7, 1e2])
plt.xlabel('frequency [kHz]',fontsize = 15)
#plt.ylabel(r'PSD kHz',fontsize = 15)
plt.title('Spectrum',fontsize = 15)
plt.ylim(10**(-8),1)
plt.xlim(0,200)
plt.yticks(fontsize = 13)
plt.xticks(fontsize = 13)
plt.legend(fontsize = 12)
plt.grid()
plt.show()