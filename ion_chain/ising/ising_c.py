# -*- coding: utf-8 -*-
"""
Compute the complete Hamiltonian for the ising coupling system
function: Htot
@author: zhumj
"""
import numpy as np
from qutip import *
import Qsim.ion_chain.ising.ising_ps as iscp
import Qsim.operator.spin as spin
import Qsim.ion_chain.transfer.exci_operators as exop
import Qsim.ion_chain.interaction.spin_phonon as Isp
from  Qsim.ion_chain.ising.ion_system import *

def summary():
    print("____________________________________________________________________")
    print("function: Htot")
    print("Genearte the total Hamiltonian in the format required by the Qutip solver (string method), with ising coupling constructed only with sx and magentic field coupled with sz")
'''
subfunction
'''

'''
function to use
''' 
def H_ord(Bz,ion0):
    '''
    Genearte the total Hamiltonian in the format required by the Qutip solver (string method),
    with ising coupling constructed only with sx and magentic field coupled with sz
    input(H0,fr,fb,N,fz,fx,delta,clevel)
    Parameters
    ----------
    H0 : qutip operator
       time independent part of the Hamiltonian
    ion0: ions class object
        contains all parameters of the ion-chain system
    Returns
    -------
    Heff : list
        list of Qutip Operator and string expressions for time dependent functions, 
        format required by the Qutip solver
    Hargd : dictionary
      dictionary that records the value of coefficients for time dependent functions
    '''
    Ns = ion0.df_spin()
    H0 = tensor(iscp.HBz(ion0,Bz),exop.p_I(ion0))
    Heff = [H0]+ Isp.H_td(ion0,0,1) + Isp.H_td(ion0,1,1)
    H_arg = Isp.H_td_arg(ion0)
    return Heff, H_arg