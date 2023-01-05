# -*- coding: utf-8 -*-
"""
Compute the complete time-dependent Hamiltonian with anharmonic terms
for  3 ion open qunatum system, under resonant condition which couples
axial tilt and radial rock mode.  
The laser field is only coupled to the ion on the side.
function: Htot
@author: zhumj
"""
import numpy as np
from qutip import *
import Qsim.operator.spin as spin
import Qsim.ion_chain.ising.ising_cex as iscex
import Qsim.ion_chain.transfer.exci_operators as exop
from  Qsim.ion_chain.ising.ion_system import *
def summary():
    print("____________________________________________________________________")
    print("function: Htot")
    print("Genearte the time-dependent Hamiltonian for 1 site electron tranfer with anhormonic terms in ordinary interaction frame")

'''
function to use
''' 
def Htot(dE, ion0,df,spterm = True, ah_term=False,ah_op=0):
    '''
    Compute the complete time-dependent Hamiltonian and collapse operators for the 3 ion open qunatum system
    used to simulate excitation transition between 2 sites, and the collpase operators
    Input:
    ----------
    dE : float
       site energy difference [kHz] 
    
    ion0: ions class object
        the object that represent the system to be simulated
    df: integer
       viberational coupling direction, 0 for axial, 1 for transverse 
    an_term: bool
        if anharmonic terms will be included
    an_op: Qutip operator
        anharmonic coupling operator
    Returns
    -------
    Heff : list
        list of Qutip Operator and string expressions for time dependent functions, 
        format required by the Qutip solver, this list represents the time-dependent 
        Hamiltonian of the system in ordinary frame
    Hargd : dictionary
        dictionary that records the value of coefficients for time dependent functions
    '''
    Ns = ion0.df_spin() #of ions to be considered for spin space
    Hlistd,Hargd = iscex.Htd(ion0,0,df)
    Hlistu,Hargu = iscex.Htd(ion0,1,df)
    #phonnic mode
    pI= exop.p_I(ion0)
    #coupling between sites, flop strength
    term3 = fr_conv(ion0.Omegax,'hz') * tensor(spin.sx(Ns,0),pI)
    #site energy difference
    term4 = -0.5*fr_conv(dE,'hz') * tensor(spin.sz(Ns,0), pI) #needs to be positive for effective transfer
    if ah_term:
        term5 = ah_op + ah_op.dag()
    else:    
        term5 = 0
    H0 = term3+term4+term5
    Heff = [H0] + Hlistd + Hlistu 
    if spterm:
        return Heff, Hargd
    else:
        return H0