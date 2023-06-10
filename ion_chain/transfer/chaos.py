# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 20:07:26 2023
Compute average ratio of adjacent energy level gaps to quantify chaotic behavior
that could be induced by the structure of Hamiltonian
reference: 
@author: zhumj
"""
from qutip import *
import matplotlib.pyplot as plt
import Qsim.operator.spin as spin
import Qsim.operator.spin_phonon as sp_op
import numpy as np
import datetime
def create_dic(data):
    #convert the data list to frequency distribution
    dist = {}
    for i in range(len(data)):
        dist[str(data[i])]=dist.get(str(data[i]),0)+1
    return dist
def bin_data(data,length):
    return (np.floor_divide(data,length))*length
 
def AE_ratio(H,plot_sp=False,printt=False):
    '''
    Compute average ratio of adjacent energy level gaps for an input Hamiltonian
    Parameters
    ----------
    plot_sp: bool 
        if true, plot the spectrum of input H 
    H : Qutip operator
        Time independent Hamiltonian as an input    
    Returns
    -------
    float, average ratio of adjacent energy level gaps
    0.53 for a Wigner–Dyson distribution and 0.39 
    in the Poissonian case
    '''
    #diagoniaze H
    if printt:
        print('Computing Eigenenergies')
        start_t = datetime.datetime.now()
    earray = np.unique(H.eigenenergies())
    if printt:
        end_t = datetime.datetime.now()
        elapsed_sec = (end_t - start_t).total_seconds()
        print("Computation completed, time consumed " + "{:.2f}".format(elapsed_sec) + "s")  
    if plot_sp:
        plt.figure(0)
        bined_e = bin_data(earray,H.shape[0]/100)/H.shape[0]
        val_dist = create_dic(bined_e)
        plt.bar(list(val_dist.keys()),list(val_dist.values()))
        plt.show()
    #compute r
    result = np.array([])
    for n in range(1,np.size(earray)-1):
        ne_array = np.array([earray[n+1] - earray[n], earray[n] - earray[n-1]])
        result = np.append(result, np.min(ne_array)/np.max(ne_array))
    return np.mean(result)
def L_imbalance(ini_spin,ion0):
    '''
    compute the L observable which is the difference beetween the 
    avaerage magnetization of the two groups of initial up/ down  spins

    Parameters
    ----------
    ini_spin: array of int 0,1 
        initial spin configuration of the system, 0 for up and 1 for down
    ion0 : ion class object

    Returns
    -------
    Qutip operator

    '''
    Ns = ion0.df_spin
    up_index = np.reshape(np.argwhere(ini_spin==0),-1)
    down_index =  np.reshape(np.argwhere(ini_spin==1),-1)
    s_op1 = spin.zero_op(Ns)
    s_op2 = spin.zero_op(Ns)
    if np.size(up_index) != 0:
        for j in up_index:
            s_op1 = s_op1 + spin.sz(Ns,j)
        s_op1 = s_op1/np.size(up_index)
    if np.size(down_index) != 0:
        for j in down_index:
            s_op2 = s_op2 + spin.sz(Ns,j)
        s_op2 = s_op2/np.size(down_index)
    s_op = s_op1 - s_op2
    L_op = tensor(s_op, sp_op.p_I(ion0)) 
    return L_op 