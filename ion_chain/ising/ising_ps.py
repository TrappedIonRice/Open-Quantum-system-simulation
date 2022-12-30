# -*- coding: utf-8 -*-
"""
Compute basic physical quantities of ising coupling system 
Generate the Hamiltonian under pure spin approximation
functions: lc, eta, Omega, Axialfreq, Axialmode, Transfreq, Transmode, Jt, plotj, HBz, Hps
@author: zhumj
"""
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import Qsim.operator.spin as spin
from  Qsim.ion_chain.ising.ion_system import *
'''
Define phyiscal constants of the system
'''
h = 6.62607015 * 10**(-34) / (2*np.pi)
MYb171 = 0.171 / (6.02214076*10**23) #mass of Yb ion, kg
qe = 1.60218 * 10**(-19) #charge of electron, C
eps0 = 8.85418781 * 10**(-12) #vacuum dielectric constant,SI
Dk = np.sqrt(2)*2*np.pi / (355*10**(-9)) 
#difference in wavevector projection of the 2 lasers m-1
R = (h*Dk**2) / (2*MYb171) #recoil frequency constant, SI 
def summary():
    print("____________________________________________________________________")
    print("function: Jt")
    print("Compute Ising coupling matrix J")
    print("____________________________________________________________________")
    print("function: plotj")
    print("visiualize the matrix elements in Jt")
    print("____________________________________________________________________")
    print("function: HBz")
    print("compute the Hamiltonian coupled with z magnetic field    ")
    print("____________________________________________________________________")
    print("function: Hps")
    print("Compute Hamiltonian under a pure spin approximation, with ising coupling constructed only with sx and magentic field coupled with sz")
def w(f):
    #transfer to the radial frequency in Hz
    #input fz (MHZ) 
    return 2*np.pi*10**6 * f

'''
functions to use
'''
#Compute Ising coupling matrix J
def Jt(ion0):
    '''
    Compute Ising coupling matrix J
    Input: (fr,fb,N,fz,fx,delta0)
    Parameters
    ----------
    ion0: ions class object
        contains all parameters of the ion-chain system
    Returns
    -------
    np array object that represents N by N matrix J

    '''
    fr = ion0.fr; fb=ion0.fb
    N = ion0.N; delta0 = ion0.delta
    fz = ion0.fz; fx = ion0.fx
    wx = w(fx); wz = w(fz); sdelta = delta0*2*np.pi*10**3
    Omegar = Omega(fr,fx); Omegab = Omega(fb,fx)
    Omega0 = Omegar*Omegab
    print(Omega0)
    nfreq = ion0.Transfreq();emat = ion0.Transmode()
    J = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i != j:
                eij = 0
                for m in range (N):
                    numer = R*Omega0 * emat[m,i] * emat[m,j]
                    demon = (sdelta+wx)**2 - (wz*nfreq[m])**2
                    eij = eij + (numer/demon)
                J[i,j] = eij/(2*np.pi*10**3)    
    return J
def plotj(J):
    '''
    visiualize the matrix elements in J
    Parameters
    ----------
    J : np array
        np array object that represents N by N matrix J, output of function Jt

    Returns
    -------
    None.

    '''
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(121, projection='3d')
    ndim = np.shape(J)[0]
    plot_c = np.arange(1,ndim+1)
    xx, yy = np.meshgrid(plot_c,plot_c)
    xcoord = xx.ravel(); ycoord = yy.ravel()
    ax1.bar3d(xcoord, ycoord, np.zeros_like(xcoord), 1, 1, J.ravel(), shade=True)
    ax1.set_title('Coupling matrix J')
    ax1.set_xlabel('i index')
    ax1.set_ylabel('j index')
def HBz(N,B0):
    '''
    compute the Hamiltonian due to coupling with z magnetic field
    input(N,B0)
    Parameters
    ----------
    N : int
        number of ions in the system
    B0 : float
        effective field strength

    Returns
    -------
    Qutip operator

    '''   
    H = spin.zero_op(N)
    for i in range(N):
        H = H + B0 * spin.sz(N,i) 
    return 2*np.pi*H    
def Hps(J,N,B0):
    '''
    Compute Hamiltonian under a pure spin approximation, with ising coupling constructed only with sx and magentic field coupled with sz
    input(J,N,B0)
    Parameters
    ----------
    J : np array
        np array object that represents N by N matrix J, output of function Jt
    N : int
        number of ions in the system    
    B0 : float
        effective field strength

    Returns
    -------
    Qutip operator

    ''' 
    H = spin.zero_op(N)
    for i in range(1,N):
        submat = spin.zero_op(N)
        for j in range(i):
            submat = submat + J[i,j]*spin.sx(N,i)*spin.sx(N,j)
        H = H + submat
    return 2*np.pi*H +  HBz(N,B0)
 