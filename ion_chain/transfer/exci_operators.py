# -*- coding: utf-8 -*-
"""
operators used to construct Hamiltonian for excitation transfer

@author: zhumj
"""

import Qsim.operator.phonon as phon
from  Qsim.ion_chain.ising.ion_system import *
def ph_list(ion0,df):
    if ion0.df_phonon() [0]== 1: #only consider one phonon space
        mlist = ion0.active_phonon[0]
    else:   #two  phonon spaces
       mlist = ion0.active_phonon[df]
    return mlist    
def pnum(ion0,df=1):
    '''
    find the number of phonon spaces coupled to the laser
    
    Parameters
    ----------
    ion0 : ion class object
    df : int
        vibrational degree of freedom that couples to the laser, 0: axial, 1: radial

    Returns
    -------
    None.

    '''
    if ion0.df_phonon() [0]== 1: #only consider one phonon space
        dim = ion0.df_phonon()[1][0]
    else:   #two  phonon spaces
        dim = ion0.df_phonon()[1][df]
    return dim    
def p_zero(ion0):
    '''
    construct the zero operator on phonon space
    Parameters
    ----------
    ion0 : ion class object
    df : int
        vibrational degree of freedom that couples to the laser, 0: axial, 1: radial

    Returns
    -------
    None.

    '''
    Np = pnum(ion0)
    pcut =ion0.pcut
    if ion0.df_phonon()[0] == 1: #only consider one phonon space
        pzero = phon.zero_op(pcut,Np)
    else:     #two  phonon spaces
        pzero = tensor(phon.zero_op(pcut[0],ion0.df_phonon()[1][0]),
                       phon.zero_op(pcut[1],ion0.df_phonon()[1][1]))
    return pzero  
def p_I(ion0):
    '''
    construct the identity operator on phonon space
    Parameters
    ----------
    ion0 : ion class object
    df : int
        vibrational degree of freedom that couples to the laser, 0: axial, 1: radial

    Returns
    -------
    None.

    '''
    Np = pnum(ion0)
    pcut =ion0.pcut
    if ion0.df_phonon()[0] == 1: #only consider one phonon space
        pI = phon.pI(pcut,Np)
    else:     #two  phonon spaces
        pI = tensor(phon.pI(pcut[0],ion0.df_phonon()[1][0]),
                    phon.pI(pcut[1],ion0.df_phonon()[1][1]))
    return pI
def p_ladder(ion0,df,mindex,atype):
    '''
    construct the identity operator on phonon space
    Parameters
    ----------
    ion0 : ion class object
    df : int
        vibrational degree of freedom that couples to the laser, 0: axial, 1: radial
    mindex: int  
        index of phonon space where the ladder operator is acting on    
    atype: int 
        type of phonon operator, 0 for down, 1 for up
    Returns
    -------
    None.

    ''' 
    Np = pnum(ion0,df)
    pcut = ion0.pcut
    if ion0.df_phonon()[0] == 1: #only consider one phonon space
        if atype == 0:
            opa = phon.down(mindex,pcut,Np)
        else:
            opa = phon.up(mindex,pcut,Np)
    else:     #two  phonon spaces
        if atype == 0:
            opa = phon.down(mindex,pcut[df],Np)
        else:
            opa = phon.up(mindex,pcut[df],Np)
        #construct in order axial, transverse
        if df ==0:
            opa = tensor(opa,phon.pI(pcut[1],ion0.df_phonon()[1][1]))
        else:
            opa = tensor(phon.pI(pcut[0],ion0.df_phonon()[1][0]),opa)
    return opa    