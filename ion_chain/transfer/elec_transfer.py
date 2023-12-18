# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 15:47:46 2022

@author: zhumj
Construct Hamiltonian of a 2 ion open qunatum system used to simulate electron transition between acceptor and donor state
in a single site system.
"""

import numpy as np
import Qsim.ion_chain.interaction.spin_phonon as Isp
import Qsim.ion_chain.interaction.pure_spin as Is
from  Qsim.ion_chain.ion_system import *
from qutip import *
#subfunction
def summary():
    '''
    give a summary of all functions and classes defined in this module
    '''
    print('___________________________________________________________________')
    print('U')
    print('compute the activation energy to reasonance conformation')
    print('___________________________________________________________________')
    print('H_res')
    print('Generate time-independent Hamiltonian for 2 state electron transfer system in resonant interaction frame')
    print('___________________________________________________________________')
    print('H_ord')
    print('Generate the time-dependent Hamiltonian for 2 state electron transfer system in ordinary interaction frame')

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
def H_res( ion0, laser0, Omegax=0, Omegay=0,Omegaz=0, i_type=0,normalized=False):
    '''
    Genearte time-independent Hamiltonian for 2 state electron transfer system in resonant interaction frame

    Parameters
    ----------
    Omegax : float 
        coupling coefficient between the doner and acceptor state [kHz]
    Omegaz : float
        energy difference between the doner and acceptor state  [kHz]
    ion0 : ion class object
    laser0: laser class object
    noramlized: bool
         if True, normalize the coefficient with the corresponding eigenmode index
    Returns
    -------
    Heff: Qutip operator
        Effective Hamiltonian in resonant frame
    '''
    H_s =  Is.single_site(ion0,Omegax, Omegay,Omegaz)
    Heff = H_s+ Isp.H_res(ion0, laser0, i_type, normalized)
    return Heff
def H_ord(Omegax, Omegay,Omegaz, ion0, laser0, i_type=0):  ## Included Omegay
    '''
    Generate the time-dependent Hamiltonian for 2 state electron transfer system in ordinary interaction frame,
    in the format required by the Qutip solver (string method) 

    Parameters
    ----------
    Omegax : float 
        coupling coefficient between the doner and acceptor state [kHz]
    Omegaz : float
        energy difference between the doner and acceptor state  [kHz]
    ion0 : ion class object
    laser0: laser class object
    Returns
    -------
    Heff: list
        time-dependent Hamiltonian list required by the solver
    H_arg: dictionary
        dic of argument parameters
    '''
    H_s =  Is.single_site(ion0, Omegax, Omegay, Omegaz)  ## Included Omegay and switch order of ion0
    Heff = [H_s] + Isp.H_td(ion0,laser0,0, i_type) + Isp.H_td(ion0,laser0,1, i_type)
    H_arg = Isp.H_td_arg(ion0,laser0)
    return Heff, H_arg

def multi_E_task(H, rho,t_array,clist,elist):
    #task function to be used for parallel computing
    #rho = s_para['rho']; clist = s_para['clist'];
    #t_array = s_para['t_array']; elist = s_para['elist'];
    result = mesolve(H,rho,t_array,clist,elist,options=Options(nsteps=100000))
    rhoee = result.expect[0]
    return rhoee 

def multi_H_parallel(sim_para,Hlist,n_cpu):
    '''
    Simulate electron transfer given a list of Hamiltonian using parallel computing
    
    Parameters
    ----------
    sim_para : dict 
        A dictionary that takes the following format:
            {
             'rho' : , (initial density matrix)
             't_array':, (time array for sampling dynamics)
             'elist' : , (list of observables to be evaluated)
             'clist' :  (list of collapse operators to construct Lindbladian operator)
             }
    Hlist : list of qutip operators
        the list of H to be simulated 
    n_cpu: int
        number of cpu to be used for simluation 
    Returns
    -------
    p_result, list of all simulation results

    '''
    rho_s = sim_para['rho'] ; t_array_s = sim_para['t_array'] ; 
    clist_s = sim_para['clist'] ; elist_s = sim_para['elist'] ; 
    p_result = parallel_map(multi_E_task, values = Hlist,task_args=True,
                            progress_bar=True,num_cpus=n_cpu,
                            rho=rho_s,t_array=t_array_s,clist=clist_s ,elist=elist_s)
    return p_result
    
    
