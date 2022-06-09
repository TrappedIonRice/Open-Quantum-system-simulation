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
#%%
'''
    parameters of the system
'''    
N = 2; #number of ions
fa = 0.2; #axial COM (Confining) frequency MHz
ft = 5; #transverse COM (Confining) frequency MHz
fsb = 30 #side band rabi frequency kHz
fzz = 10 #ZigZag detuning frequency, kHz 
Bz = 0 #Effective magnetic field
delta = float(input("Enter detuning frequency (kHz): "))
def printpar():
    print('number of ions', N)
    print('axial COM (Confining) frequency ',fa,' MHz')
    print('transverse COM (Confining) frequency ',ft, 'MHz')
    print('side band rabi frequency ',fsb, ' kHz')
    print('ZigZag detuning frequency', fzz,' kHz')
    print('detunning frequency',np.round(delta,1), 'kHz')
printpar()    
'''
simulation for complete Hamiltonian
'''
pcut = 3 #truncation of phonon energy
#construct Hamiltonian 
#time indepenedent part
H0 = tensor(iscp.HBz(N,Bz),phon.pI(pcut,N))
Heff,arg0 = iscc.Htot(H0,fsb,fsb,N,fa,ft,delta,pcut)
#construct initial state
nlist0 = [0,0,0]
psi1 = tensor(spin.phid(N),phon.phip(pcut,N,nlist0))
elist1 = [tensor(spin.sz(N,0),phon.pI(pcut,N)),tensor(spin.sz(N,1),phon.pI(pcut,N))]
#solve time dependent SE
times =  np.arange(0,1,10**(-4))
print('______________________________________________________________________')
print('solving for complete Hamiltonian')
result1 = mesolve(Heff,psi1,times,e_ops=elist1,args = arg0,progress_bar=True,options=Options(nsteps=1000))    
#%%
psi0 = spin.phid(N)  
J = iscp.Jt(fsb,fsb,N,fa,ft,delta)
elist2 = [spin.sz(N,0),spin.sz(N,1)]
H = iscp.Hps(J,N,Bz)
print('______________________________________________________________________')
print('solving for pure spin interaction')
result = mesolve(H,psi0,times,e_ops=elist2,progress_bar=True, options=Options(nsteps=1000))
#%%
plt.plot(times,result.expect[0]+result.expect[1],label = 'spin')
plt.plot(times,result1.expect[0]+result1.expect[1],label = 'complete')
plt.xlabel(r'$t$ [ms]')
title = 'detuning '+str(delta)+' kHz'
plt.ylabel(r'$<\sigma_{zz}>$')
plt.title(title)
plt.legend()
plt.grid()
