# -*- coding: utf-8 -*-
"""
Compute ion-laser interaction Hamiltonian in resonant/ordinary interaction frame
@author: zhumj
"""
import numpy as np
from qutip import *
import Qsim.operator.spin as spin
import Qsim.operator.phonon as phon
import Qsim.ion_chain.transfer.exci_operators as exop
from  Qsim.ion_chain.ion_system import *
def summary():
    print("____________________________________________________________________")
    print("function: H_td")
    print("Genearte time-dependent Hamiltonian for laser-ion interaction in ordinary frame")
    print("____________________________________________________________________")
    print("function: H_td_arg")
    print(" Generate an argument dictonary which maps parameters in time-dependent expressions to their actual values")
    print("____________________________________________________________________")
    print("function: H_res")
    print("Genearte time-independent Hamiltonian for laser-ion interaction in resonant frame")
    
'''
subfunctions
'''    
def sigma_phi(N,i,phase):
    return np.cos(phase)*spin.sx(N,i) + np.sin(phase)*spin.sy(N,i)

def tstring(N,atype,las_label=''):
    '''
    Generate the list of time depedent expression for the Hamiltonian 

    Parameters
    ----------
    N : int
        total number of ions in the trapped ion system
    atype : int
        type of phonon operators, 0 for down, 1 for up
    las_label: str, default as ''
        extra label for the laser drive, specify when using more than 1 laser drives
    Returns
    -------
    mstring : list of string
        list of parameters
    fstring : list of string
        list of time dependent expressions to be used 

    '''
    mstring = []
    fstring = []
    ustr = 'u'+las_label
    for mi in range(1,N+1):
        newm = "m" + las_label + str(mi)
        mstring.append(newm)
        if atype == 1:
            fstring.append('cos(t * '+ustr+') * exp(t * ' + newm +")")
        else:
            fstring.append('cos(t * '+ustr+') * exp(-1 * (t * ' + newm +"))")
    return mstring, fstring        

def Him_ord(ion0,atype,i,m,sindex,mindex,i_type=0):
    '''
    Compute the i,m th component for time independent part of ion-laser interaction 
    Hamiltonian in ordinary frame, which discribes the coupling between ion i and mode m
    Input: 
        ion0: ion class object
        atype: int
            phonon opeartor type, 0 for destroy, 1 for create
        i: int
            ion index 
        m: int
            phonon space index
        sindex: int
            index to construct spin operator
        mindex: int
            index to construct phonon operator
        i_type: int default as 0
            type of interaction, set to 1 for ising interactions
            
    Output:
        Hamiltonina H im, Qobj
    '''
    #set coefficient constants according to the coupling degree of freedom
    p_opa = exop.p_ladder(ion0,mindex,atype)
    if i_type == 1:
        s_oper = sigma_phi(ion0.df_spin(),sindex,ion0.phase)
    else:    
        s_oper = spin.sz(ion0.df_spin(),sindex)
    H = tensor(s_oper,p_opa)
    return ion0.g(i,m)*H 

def Him_res(ion0, i,m,sindex,mindex):
    '''
    Compute the i,m th component for ion-laser interaction  Hamiltonian in resonant frame, 
    which discribes the coupling between ion i and mode m
    Input: 
        ion0: ion class object
        i: int
            ion index 
        m: int
            phonon space index
        sindex: int
            index to construct spin operator
        mindex: int
            index to construct phonon operator
    Output:
        Hamiltonina H im, Qobj
    '''
    #set coefficient constants according to the coupling degree of freedom
    p_opa = exop.p_ladder(ion0,mindex,0) + exop.p_ladder(ion0,mindex,1) 
    H = tensor(spin.sz(ion0.df_spin(),sindex),p_opa)
    return 0.5*ion0.g(i,m)*H 

'''
functions to use
''' 

def H_td_arg(ion0,las_label=''):    
    '''
    Generate an argument dictonary which maps parameters in time-dependent 
    expressions to their actual values
    Parameters
    ----------
    ion0: ion class object
    las_label: str, default as ''
        extra label for the laser drive, specify when using more than 1 laser drives
    Returns
    -------
    adic : dictionary
        argument dictonary
    '''
    #generate the arg list for solving time dependent SE
    #wlist is the list of eigenfrequencies, mu is the frequency of the laser
    adic = {"u"+las_label:ion0.mu()}
    slist, fs = tstring(ion0.N,0,las_label)
    wlist0 = 1j*ion0.wmlist() * 2000* np.pi #this is used to compute deltam in kHz
    for argi in range(ion0.N):
        adic[slist[argi]] = wlist0[argi]
    return adic 

def H_td(ion0,atype,i_type = 0,las_label=''): 
    '''
    Compute the list of H correponding to time-dependent Hamiltonian for ion-lasesr
    interaction with a single drive as a input for qutip solver
    Input: 
        ion0, ion class object
        atype: int
            phonon opeartor type, 0 for destroy, 1 for create
        i_type: int default as 0
            type of interaction, set to 1 for ising interactions 
        las_label: str, default as ''
            extra label for the laser drive, specify when using more than 1 laser drives    
    '''
    Hstr, Hexpr = tstring(ion0.N,atype,las_label) #kHz generate time depedent part for all modes and select 
                                      # modes of interest           
    #compute the mth element by summing over i for Him for destroy operators
    Hlist = []
    mindex = 0 #this index is used for phonon operators
    for m in exop.ph_list(ion0):
        sindex = 0 #this index is used for spin operators
        subH = tensor(spin.zero_op(ion0.df_spin()),exop.p_zero(ion0))
        for i in ion0.laser_couple:
            subH = subH + Him_ord(ion0,atype,i,m,sindex,mindex,i_type)
            sindex = sindex + 1
        mindex = mindex+1
        Hlist.append([subH,Hexpr[m]]) 
    return Hlist

def H_res(ion0):
    '''
    Compute the time-independent Hamiltonian e for ion-lasesr
    interaction with a single drive in resonant fram
    Input: 
        ion0, ion class object
    '''
    term1 = tensor(spin.zero_op(ion0.df_spin()),exop.p_zero(ion0)) #laser-ion interaction term 
    term2 = tensor(spin.zero_op(ion0.df_spin()),exop.p_zero(ion0)) #compensation for change of interaction frame
    mindex = 0 #this index is used for phonon operators
    for m in exop.ph_list(ion0):
        sindex = 0 #this index is used for spin operators
        for i in ion0.laser_couple:
            term1 = term1 + Him_res(ion0,i,m,sindex,mindex)
            sindex = sindex + 1
        term2 = (term2 + ion0.dmlist()[m]
                 *tensor(spin.sI(ion0.df_spin()),
                         exop.p_ladder(ion0,mindex,1)*exop.p_ladder(ion0,mindex,0)))    
        mindex = mindex+1
    return term1-term2
        