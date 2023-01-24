# -*- coding: utf-8 -*-
"""
Compute the complete time-dependent Hamiltonian with anharmonic terms
for  3 ion open qunatum system
The laser field is only coupled to the ion on the side.
function: Htot
@author: zhumj
"""
import numpy as np
from qutip import *
import Qsim.ion_chain.transfer.exci_operators as exop
import Qsim.ion_chain.interaction.spin_phonon as Isp
import Qsim.ion_chain.interaction.pure_spin as Is
from  Qsim.ion_chain.ion_system import *
def summary():
    print("____________________________________________________________________")
    print("function: H_ord")
    print("Genearte the time-dependent Hamiltonian for 1 site electron tranfer with anhormonic terms in ordinary interaction frame")

'''
subfunction
'''
def ord_str(N,dtype):
    '''
    Generate a list of time depedent expression for the anharmonic Hamiltonian
    in ordinary interaction frame, in form exp(dmnp * t) exp(-dmnp * t)
    Parameters
    ----------
    N: int
        Number of ions in the system
    dtype : int
        type of anharmonic term, 0 for +, 1 for -

    Returns
    -------
    None
    '''
    return None
def res_str(dtype,sign):
    '''
    Generate time depedent expression for the anharmonic Hamiltonian
    in resonant interaction frame, in form exp(+- d1 * t) exp(+-d2 * t)
    where d1 = -j * mu_1, d2 = j(2*mu_2-mu_1)
    Parameters
    ----------
    dtype : int
        type of anharmonic term, 0 for d1, 1 for d2
    sign : int
        sign of the exponent, 0 for - . 1  for +, assume negative for adjoint operators

    Returns
    -------
    str, time depedent expression for the anharmonic Hamiltonian

    '''
    if dtype == 0:
        tfactor ='d1'
    else:
        tfactor = 'd2'
    if sign == 0:
        texp = 'exp(-1 * (t * ' + tfactor +"))"
    else:
        texp = 'exp(t * ' + tfactor +')'
    return texp
        
def res_arg(mu1,mu2):
    '''
    Generate an argument dictonary which maps parameters in time-dependent 
    expressions for anharmonic term in resonant frame to their actual values

    Parameters
    ----------
    mu1 : float
        side-band frequency of axial laser [2pi kHz]
    mu2 : TYPE
        side-band frequency of radial laser [2pi kHz]

    Returns
    -------
    dictionary for argument mapping

    '''
    return {'d1':-1j*mu1,'d2':1j*(2*mu2-mu1)}

'''
function to use
''' 
def H_ord1(Omegax, Omegaz, ion0,sp_term = True, ah_term=False,ah_op=0):
    '''
    Compute the complete time-dependent Hamiltonian for the 3 ion open qunatum system
    used to simulate electron of a single site with anharmonicity in ordinary interaction frame with a single
    laser drive
    Input:
    ----------
    Omegax : float 
        coupling coefficient between the doner and acceptor state [kHz]
    Omegaz : float
        energy difference between the doner and acceptor state  [kHz]
    ion0: ions class object
        the object that represent the system to be simulated
    sp_term: bool:
        if spin-phonon interacion term will be included
    an_term: bool
        if anharmonic terms will be included
    an_op: Qutip operator
        anharmonic coupling operator with coupling strength
    Returns
    -------
    Heff : list
        list of Qutip Operator and string expressions for time dependent functions, 
        format required by the Qutip solver, this list represents the time-dependent 
        Hamiltonian of the system in ordinary frame
    Hargd : dictionary
        dictionary that records the value of coefficients for time dependent functions
    '''
    #phonnic mode
    H_s =  Is.single_site(Omegax, Omegaz, ion0) 
    if ah_term:
        term_a = ah_op + ah_op.dag()
    else:    
        term_a = 0
    H0 = H_s+term_a
    Heff = [H0] + Isp.H_td(ion0,0) + Isp.H_td(ion0,1)
    H_arg = Isp.H_td_arg(ion0)
    if sp_term:
        return Heff, H_arg
    else:
        return H0
def H_ord2(Omegax, Omegaz, ion1, ion2, ah_term=False,ah_op=0):
    '''
    Compute the complete time-dependent Hamiltonian for the 3 ion open qunatum system
    used to simulate electron of a single site with anharmonicity in ordinary interaction frame with 2 laser drives
    Input:
    ----------
    Omegax : float 
        coupling coefficient between the doner and acceptor state [kHz]
    Omegaz : float
        energy difference between the doner and acceptor state  [kHz]
    ion1: ions class object
        used to construct H for laser dirve 1 (axial)
    ion2: ions class object
        used to construct H for laser dirve 2 (radial), ion1 and ion2 should have the same Hilbert Space   
    an_term: bool
        if anharmonic terms will be included
    an_op: Qutip operator
        anharmonic coupling operator with coupling strength
    Returns
    -------
    Heff : list
        list of Qutip Operator and string expressions for time dependent functions, 
        format required by the Qutip solver, this list represents the time-dependent 
        Hamiltonian of the system in ordinary frame
    Hargd : dictionary
        dictionary that records the value of coefficients for time dependent functions
    '''
    #phonnic mode
    H_s =  Is.single_site(Omegax, Omegaz, ion1) 
    if ah_term:
        term_a = ah_op + ah_op.dag()
    else:    
        term_a = 0
    H0 = H_s+term_a
    Heff = ([H0]  + Isp.H_td(ion1,0,0,'1') + Isp.H_td(ion1,1,0,'1')
            + Isp.H_td(ion2,0,0,'2') + Isp.H_td(ion2,1,0,'2'))
    H_arg = Isp.H_td_arg(ion1,'1')|Isp.H_td_arg(ion2,'2')    
    return Heff, H_arg
def H_res1(Omegax, Omegaz, ion0, ah_term=False,ah_op=0,td=False):
    '''
    Compute the complete time-dependent Hamiltonia for the 3 ion open qunatum system
    used to simulate electron of a single site with anharmonicity in resonat interaction frame with 1 laser drive
    Input:
    ----------
    Omegax : float 
        coupling coefficient between the doner and acceptor state [kHz]
    Omegaz : float
        energy difference between the doner and acceptor state  [kHz]
    ion0: ions class object
        used to construct H for laser dirve 1 (axial) 
    an_term: bool
        if anharmonic terms will be included
    an_op: Qutip operator, default as 0
        anharmonic coupling operator with coupling strength
    Returns
    -------
    Heff : list
        list of Qutip Operator and string expressions for time dependent functions, 
        format required by the Qutip solver, this list represents the time-dependent 
        Hamiltonian of the system in ordinary frame
    Hargd : dictionary
        dictionary that records the value of coefficients for time dependent functions
    '''
    H_s =  Is.single_site(Omegax, Omegaz, ion0) 
    H0 = H_s + Isp.H_res(ion0) 
    return H0
def H_res2(Omegax, Omegaz, ion1, ion2, ah_term=False,ah_op=0,td=False):
    '''
    Compute the complete time-dependent Hamiltonia for the 3 ion open qunatum system
    used to simulate electron of a single site with anharmonicity in resonat interaction frame with 2 laser drives
    Input:
    ----------
    Omegax : float 
        coupling coefficient between the doner and acceptor state [kHz]
    Omegaz : float
        energy difference between the doner and acceptor state  [kHz]
    ion1: ions class object
        used to construct H for laser dirve 1 (axial)
    ion2: ions class object
        used to construct H for laser dirve 2 (radial), ion1 and ion2 should have the same Hilbert Space   
    an_term: bool
        if anharmonic terms will be included
    an_op: Qutip operator, default as 0
        anharmonic coupling operator with coupling strength
    td: bool, default as False
        simulate resonat anharmoncity (False) or off-resonance time-dependent anharmonic terms (True)
        If true, the input ah_op should be a list of 2 operators, where the first index correpond to
        mu1 and second to 2(mu2-mu1)
    Returns
    -------
    Heff : list
        list of Qutip Operator and string expressions for time dependent functions, 
        format required by the Qutip solver, this list represents the time-dependent 
        Hamiltonian of the system in ordinary frame
    Hargd : dictionary
        dictionary that records the value of coefficients for time dependent functions
    '''
    H_s =  Is.single_site(Omegax, Omegaz, ion1) 
    if ah_term:
        if td:
            H_arg = res_arg(ion2.mu(),ion1.mu())
            ahlist = []
            for tindex in range(2):
                ahlist.append(ah_op[tindex], res_str(tindex,1))
                ahlist.append(ah_op[tindex].dag(),res_str(tindex,0))
        else: #resonant anharmonic coupling
            term_a = ah_op + ah_op.dag()
    else:    
        term_a = 0
    H0 = H_s + Isp.H_res(ion2) + Isp.H_res(ion1) 
    if td:
        Heff = [H0] + ahlist
        return Heff, H_arg
    else:
        Heff = H0 + term_a
        return Heff