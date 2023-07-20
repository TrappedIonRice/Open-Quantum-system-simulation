# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 23:27:39 2023
This module is used for simuation of spin-squeezing generation. The purpose is to
directly set the parameters without using the ion class.
@author: zhumj
"""
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import Qsim.operator.spin as spin
import Qsim.operator.phonon as phon
import sigfig
def H_res(delta=0,E=0,V=0,Omega=0,cutoff=2):
    '''
    construct the Hamiltonian for the model, consider 
    rocking mode only, the second ion is used as coolant

    Parameters
    ----------
    delta : float, optional
        harmonic energy level. The default is 0.
    E : float, optional
        sigma_y coupling. The default is 0.
    V : float, optional
        sigma_x coupling. The default is 0.
    Omega : float, optional
        spin phonon coupling. The default is 0.
    cutoff : int, optional
        phonon space cutoff. The default is 2.

    Returns
    -------
    None.

    '''
    #operators
    a_up = tensor(spin.sI(2),phon.up(m=0,cutoff=[cutoff],N=1))
    a_down = tensor(spin.sI(2),phon.down(m=0,cutoff=[cutoff],N=1))
    sigma_x1 = tensor(spin.sx(N=2,i=0),phon.pI([cutoff],1)) 
    sigma_x3 = tensor(spin.sx(N=2,i=1),phon.pI([cutoff],1)) 
    sigma_y1 = tensor(spin.sy(N=2,i=0),phon.pI([cutoff],1)) 
    sigma_y3 = tensor(spin.sy(N=2,i=1),phon.pI([cutoff],1)) 
    # harmonic term
    Hh = delta * a_up * a_down
    #spin term
    Hs = E * (sigma_y1+sigma_y3) + V * (sigma_x1+sigma_x3)
    #spin phonon coupling
    Hsp = Omega * (a_up+a_down) * (sigma_x1+sigma_x3)
    H0 = 2*np.pi* (Hh+Hs+Hsp)
    return H0
def cooling(gamma=0, nbar=0, cutoff=2):
    clist = []
    cm = tensor(spin.sI(2),phon.down(m=0,cutoff=[cutoff],N=1))
    coeff = np.sqrt(2*np.pi*gamma)
    clist.append(coeff*np.sqrt(1+nbar)*cm)
    clist.append(coeff* np.sqrt(nbar)*cm.dag())                                         
    return clist  

def init_state(spin_state, p_num, cutoff):
    ket0 = tensor(spin_state,fock(cutoff,p_num))
    rho = ket0*ket0.dag()
    return rho
def phonon_measure(cutoff):
    op = tensor(spin.sI(2),
                phon.up(m=0,cutoff=[cutoff],N=1)*phon.down(m=0,cutoff=[cutoff],N=1))
    return op
def phonon_cutoff_error(states,cutoff):
    '''
    Compute the maximum occupation of the highest allowed phonon state
    as the error for choosing a finite phonon cutoff, and plot the phonon distribution in 
    fock basis of the corresponding state. 

    Parameters
    ----------
    states : list of states (result.state)
        state at different times extracted from qutip sesolve/mesolve result
    Returns
    -------
    p_max : float
        cutoff error to the second significant digit.

    '''
    opa = tensor(spin.sI(2),phon.state_measure([cutoff],1,cutoff-1,0))
    p_high = expect(opa,states)
    max_index = np.argmax(p_high)
    p_max = sigfig.round(p_high[max_index],2)
    print('Estimated phonon cutoff error: ', p_max)
    plt.figure()
    pstate = states[max_index].ptrace([2])
    plot_fock_distribution(pstate)
    plt.yscale('log')
    plt.ylim(np.min(pstate.diag())/10,10*np.max(pstate.diag()))
    return p_max