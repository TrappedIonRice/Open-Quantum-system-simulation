# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:23:13 2022

@author: zhumj
Compute the time evolution of a pure spin interaction Hamiltonian
"""
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import ion_chain.operator.spin as spin
import ion_chain.ising.ising_ps as iscp
#%%           
N = 2; #number of ions
fa = 0.2; #axial COM (Confining) frequency MHz
ft = 5; #transverse COM (Confining) frequency MHz
fsb = 30 #side band rabi frequency kHz
fzz = 10 #ZigZag detuning frequency, kHz 
Bz = 0 #Effective magnetic field
delta = float(input("Enter detuning frequency (kHz): "))
psi0 = spin.phid(N)  
J = iscp.Jt(fsb,fsb,N,fa,ft,delta)
times = np.arange(0,1.5,0.001)
#result = sesolve(H,psi0,times)
elist2 = [spin.sz(N,0),spin.sz(N,1)]

H = iscp.Hps(J,N,Bz)
print('______________________________________________________________________')
print('solving for pure spin interaction')
result = mesolve(H,psi0,times,e_ops=elist2,progress_bar=True, options=Options(nsteps=1000))
#%%
plt.plot(times,result.expect[0]+result.expect[1],label = 'sz1')
#plt.plot(times,result.expect[1],label = 'sz2')
#plt.plot(times,result.expect[2],label = 'szz')
#plt.plot(times,result1.expect[0],label = 'complete')
plt.xlabel(r'$t$ [ms]')
title = 'detuning '+str(delta)+' kHz'
plt.ylabel(r'$<\sigma_{zz}>$')
plt.title(title)
plt.legend()
plt.grid()