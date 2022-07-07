# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 15:47:46 2022

@author: zhumj
Construct Hamiltonian in reasonate rotating frame for the 2 ion open qunatum system used to simulation electron transition between acceptor and donor state

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import ion_chain.operator.spin as spin
import ion_chain.operator.phonon as phon
from astropy.io import ascii
from astropy.table import Table
from qutip import *
import ion_chain.ising.ising_ps as iscp
#subfunction
def w(f):
    #convert frequency Hz to angular frequency rad/s
    return 2*np.pi*f
def summary():
    '''
    give a summary of all functions and classes defined in this module
    '''
    print('___________________________________________________________________')
    print('U')
    print('compute the activation energy to reasonance conformation')
    print('___________________________________________________________________')
    print('Htot')
    print('construct Hamiltonian and collpase operators of the system in the reasonant rotating frame using a single mode, or double mode system')
    print('___________________________________________________________________')
    print('rho_ini')
    print('Construct initial density matrix according to a thermal distribution')

def U(Omegaz,lambda0):
    '''
    compute the activation energy to reasonance conformation
    Parameters
    ----------
    Omegaz : float
        rabi frequency due to coupling to magenetic field, energy splitting between
        the donor and acceptor state
    lambda0 : float
       reorgonization energy, output of Lambda

    Returns
    -------
    float [J/10**6]

    '''
    return (Omegaz - lambda0)**2 / (4*lambda0)
def Htot(Omegaz, ion0 , single_mode):
    '''
    construct Hamiltonian and collpase operators of the system in the reasonant rotating frame 
    using a single mode, or double mode system
    Parameters 
    ----------
    Omegaz : float
        rabi frequency due to coupling to magenetic field, energy splitting between
        the donor and acceptor state, [kHz]
    ion0: ions class object
        the object that represent the system to be simulated
    single_mode : bool
        use COM mode only if true
    Returns
    -------
    H
        Qutip operator
    clist : list
        list of Qutip operators required by qutip solver
    '''
    dm = ion0.dmlist()
    g = ion0.g()
    if single_mode:
        term1 = (-1)*(dm[0]* 
                 tensor(spin.sI(1), phon.up(1, ion0.pcut, 1)*phon.down(1, ion0.pcut, 1)))
        #phonnic mode
        term2 = -0.5 * w(Omegaz) * tensor(spin.sz(1,0),phon.pI(ion0.pcut,1))
        #vibrational harmonic oscillator potential
        term3 = w(ion0.Omegax) * tensor(spin.sx(1,0),phon.pI(ion0.pcut,1)) 
        #coherent coupling of the donor and acceptor states
        term4 = 0.5*g[0]*tensor(spin.sz(1,0),(phon.up(1, ion0.pcut, 1)+phon.down(1, ion0.pcut, 1)))   
        #a linear potential in the reaction coordinate z
        c0 =  tensor(spin.sI(1), phon.down(1, ion0.pcut, 1))
        #collapse operator
    else:    
        termp1 = dm[0]*phon.up(0, ion0.pcut, ion0.N)*phon.down(0, ion0.pcut, ion0.N)
        termp2 = dm[1]*phon.up(1, ion0.pcut, ion0.N)*phon.down(1, ion0.pcut, ion0.N)
        term1 =  tensor(spin.sI(1),-(termp1+termp2))
        #phonnic mode
        term2 = -0.5 * w(Omegaz) * tensor(spin.sz(1,0),phon.pI(ion0.pcut,ion0.N))
        #vibrational harmonic oscillator potential
        term3 = w(ion0.Omegax) * tensor(spin.sx(1,0),phon.pI(ion0.pcut,ion0.N)) 
        #coherent coupling of the donor and acceptor states
        termp3 = g[0]*tensor(spin.sz(1,0),(phon.up(0, ion0.pcut, ion0.N)+phon.down(0, ion0.pcut, ion0.N))) 
        termp4 = g[1]*tensor(spin.sz(1,0),(phon.up(1, ion0.pcut, ion0.N)+phon.down(1, ion0.pcut, ion0.N))) 
        term4 = 0.5*(termp3+termp4)
        #a linear potential in the reaction coordinate z
        c0 =  tensor(spin.sI(1), phon.down(0, ion0.pcut, ion0.N))
        #collapse operator
    H = (term1+term2+term3+term4) 
    clist = []
    clist.append(np.sqrt(w(ion0.gamma)*(1+ion0.n_bar()))*c0)
    clist.append(np.sqrt(w(ion0.gamma)*ion0.n_bar())*c0.dag())
    return H, clist
def rho_ini(ion0,single_mode):
    '''
    Construct initial density matrix according to a thermal distribution

    Parameters
    ----------
    ion0: ions class object
       the object that represent the system to be simulated
    single_mode : bool
       use COM mode only if true

    Returns
    -------
    Qutip operator

    '''
    ini_sdm = Qobj([[1,0], [0,0]])
    if single_mode:
        rho0 = tensor(ini_sdm,phon.inip_thermal(ion0.pcut,1000*w(ion0.wmlist()[0]),ion0.Etot))
    else:
        pho0 = tensor(phon.inip_thermal(ion0.pcut,1000*w(ion0.wmlist()[0]),ion0.Etot),
                      phon.inip_thermal(ion0.pcut,1000*w(ion0.wmlist()[1]),ion0.Etot))
        rho0 = tensor(ini_sdm,pho0)
    return rho0    
        
    
    
