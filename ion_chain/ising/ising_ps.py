# -*- coding: utf-8 -*-
"""
Generate the Hamiltonian for ising coupling system under pure spin approximation
@author: zhumj
"""
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import Qsim.operator.spin as spin
from  Qsim.ion_chain.ion_system import *
'''
Define phyiscal constants of the system
'''

def summary():
    print("____________________________________________________________________")
    print("function: Jt")
    print("Compute ising coupling matrix J for pure spin approximation")
    print("____________________________________________________________________")
    print("function: plotj")
    print("visiualize the matrix elements in Jt")
    print("____________________________________________________________________")
    print("function: HBz")
    print("compute the Hamiltonian coupled with z direction magnetic field    ")
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
def Jt(ion0,laser0):
    '''
    Compute ising coupling matrix J for pure spin approximation
    Parameters
    ----------
    ion0: ions class object
        contains all parameters of the ion-chain system
    laser0: laser class object
        represents a symmetric pair of sideband drives
    Returns
    -------
    np array object that represents N by N matrix J, unit of [kHz]

    '''
    N = ion0.df_spin; 
    Omega0 = (laser0.Omega(ion0)*1000)**2
    emat = ion0.radial_mode
    J = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i != j:
                eij = 0
                for m in range (N):
                    numer = laser0.R * Omega0 * emat[m,i] * emat[m,j]
                    demon = 1e6*((fr_conv(laser0.mu,'hz'))**2 - (fr_conv(efreq(ion0,laser0)[m],'khz'))**2)
                    eij = eij + (numer/demon)
                J[i,j] = eij/(2*np.pi*10**3)    
    return J
def plotj(J):
    '''
    visiualize the matrix elements in J
    Parameters
    ----------
    J : np array
       ising coupling matrix, output of function Jt

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
def HBz(ion0=None, B0=0):
    '''
    compute the Hamiltonian due to coupling with z magnetic field
    input(N,B0)
    Parameters
    ----------
    ion0: ions class object
    B0 : float
        effective field strength in z direction [kHz]

    Returns
    -------
    Qutip operator

    '''   
    Ns = ion0.df_spin
    H = spin.zero_op(Ns)
    for i in range(Ns):
        H = H + B0 * spin.sz(Ns,i) 
    return 2*np.pi*H    
def Hps(J=np.array([]),ion0=None,B0=0):
    '''
    Compute Hamiltonian under a pure spin approximation, with ising coupling (sx) and magentic field coupled with sz
    input(J,ion0,B0)
    Parameters
    ----------
    J : np array
        ising coupling matrix, output of function Jt[kHz]
    ion0: ions class object
    B0 : float
        effective field strength in z direction [kHz]

    Returns
    -------
    Qutip operator

    ''' 
    Ns = ion0.N
    H = spin.zero_op(Ns)
    for i in range(1,Ns):
        submat = spin.zero_op(Ns)
        for j in range(i):
            submat = submat + J[i,j]*spin.sx(Ns,i)*spin.sx(Ns,j)
        H = H + submat
    return 2*np.pi*H + HBz(ion0,B0)
 