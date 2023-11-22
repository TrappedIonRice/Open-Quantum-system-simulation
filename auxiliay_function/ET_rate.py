# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 20:53:19 2023
Compute ET rate based on theoretical models
@author: zhumj
"""
from qutip import *
import Qsim.operator.spin as spin
import Qsim.operator.phonon as phon
import numpy as np
import matplotlib.pyplot as plt

def FC_matrix(cutoff,gf):
    #compute Franck-Condon matrix in fock state representation
    a = phon.down(0,[cutoff],1)
    aop = (gf*(a.dag() - a)).expm()
    return aop
def FC_sum(cutoff,E_split,gf,nbar,state_type='thermal'):
    '''
    Compute the sum of FC coefficient for a given energy splitting,
    weighted by thermal distribution 

    Parameters
    ----------
    cutoff : int
        cutoff of SHO operator
    E_split : float
        effective energy splitting
    gf : float
        effective s-p coupling
    nbar: float
        average phonon number of the system
    state_type: str
        specify the type of initial state, 
         can be 'thermal' or 'fock'
    Returns
    -------
    Float

    '''
    if state_type == 'thermal':
        pdist = phon.p_thermal(cutoff,nbar)
    elif state_type == 'fock':
        pdist = np.abs(np.array(fock(cutoff,nbar)))
    else:
        print('Incorrect specification of state type, allowed types are thermal, fock')
        return 0
    aop = FC_matrix(cutoff,gf)
    result = 0
    for i in range(cutoff):
        if i + E_split >= cutoff:
            break
        result +=  pdist[i]*(np.abs(aop[i,i + E_split]))**2
    return result
def ET_rate_Fermi(cutoff,E_split,g_fac,V_fac,nbar,state_type='thermal'):
    '''
    Compute normalized electron transfer rate 2pi k / omega_0
    based on Fermi golden rule, assuming V<<gamma.
    This model assumes Delta E = n omega_0
    g,V factors are in terms of hbar\omega
    Parameters
    ----------
    cutoff : int
        cutoff of SHO operator
    E_split : int
        Energy splitting factor, equals the number of 
        SHO energy levels required to compensate the energy gap
    gf : float
        effective s-p coupling factor
    Vf : float
        Site coupling factor, coefficient for sigma_x
    nbar : float
        Initial average phonon number
    state_type: str
        specify the type of initial state, 
         can be 'thermal' or 'fock'
    Returns
    -------
    Float
    '''
    if state_type == 'fock' and not(isinstance(nbar,int)):
        print('fock state phonon number must be integer')
        return 0
    else:
        return (2*np.pi)**2*V_fac**2*FC_sum(cutoff,E_split,g_fac,nbar,state_type)
def plot_ET_rate_Fermi(E_start,E_end,p_cutoff,g_fac,V_fac,nbar,state_type='thermal'):
    '''
    Plot normalized electron transfer rate 2pi k / omega_0
    based on Fermi golden rule for a set of energy splittings, 
    assuming V<<gamma.
    This function also assumes Delta E = n omega_0
    g,V factors are in terms of hbar\omega

    Parameters
    ----------
    E_start : int
        starting point of energy splitting, equals the number of 
        SHO energy levels required to compensate the energy gap
    E_end : int
        ending point of energy splitting, equals the number of 
        SHO energy levels required to compensate the energy gap
    cutoff : int
        cutoff of SHO operator
    gf : float
        effective s-p coupling factor
    Vf : float
        Site coupling factor, coefficient for sigma_x
    nbar : float
        Initial average phonon number
    state_type: str
        specify the type of initial state, 
         can be 'thermal' or 'fock'
    Returns
    -------
    None.

    '''
    if state_type == 'fock' and not(isinstance(nbar,int)):
        print('fock state phonon number must be integer')
        return 0
    else:
        rate_list = []
        splot = np.arange(E_start, E_end,1)
        for s in splot:
            rate_list.append(ET_rate_Fermi(p_cutoff,s,g_fac,V_fac,nbar,state_type))
        fig = plt.figure(figsize=(8, 6))
        plt.plot(splot,rate_list,'*',markersize=12)
        plt.xlabel(r'$\Delta E [\hbar\omega_0]$',fontsize = 16)
        plt.ylabel(r'$2 \pi k / \omega_0$',fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.xticks(np.arange(0, 13,1),fontsize = 16)
        #plt.xlim(1,12)
        #plt.legend(fontsize = 15)
        plt.grid()
        plt.show()