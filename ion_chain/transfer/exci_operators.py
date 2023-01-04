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
        vibrational degree of freedom on which the operator is acting, 0: axial, 1: radial
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
def rho_ini(ion0):
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
    Ns = ion0.df_spin()
    if Ns == 1:
        isket = fock(2,0)
    else:    
        isket = tensor(fock(2,0),fock(2,1)) # ion 1 in excited state
    ini_sdm = isket*isket.dag()
    if ion0.df_phonon()[0] == 1: #only consider one phonon space
        for m in range(ion0.df_phonon()[1][0]):
            if m == 0:
                pho = phon.inip_thermal(ion0.pcut[0],fr_conv(ion0.fx,'khz'),ion0.Etot)
            else:
                pho = tensor(pho,phon.inip_thermal(ion0.pcut[m],fr_conv(ion0.fx,'khz'),ion0.Etot))
    else:
        for m in range(ion0.df_phonon()[1][0]):
            if m == 0:
                pho1 = phon.inip_thermal(ion0.pcut[0][0],fr_conv(ion0.fz,'khz'),ion0.Etot)
            else:
                pho1 = tensor(pho1,phon.inip_thermal(ion0.pcut[0][m],fr_conv(ion0.fz,'khz'),ion0.Etot))
        for m in range(ion0.df_phonon()[1][1]):
            if m == 0:
                pho2 = phon.inip_thermal(ion0.pcut[1][0],fr_conv(ion0.fx,'khz'),ion0.Etot)
            else:
                pho2 = tensor(pho2,phon.inip_thermal(ion0.pcut[1][m],fr_conv(ion0.fx,'khz'),ion0.Etot))        
    #dmat = fock(ion0.pcut,0)*fock(ion0.pcut,0).dag()
    #pho0 = tensor(dmat,dmat,dmat)
        pho = tensor(pho1,pho2)
    rho0 = tensor(ini_sdm,pho)
    return rho0    
def ini_zero(ion0,state_type):
    '''
    Construct initial density matrix that has 0 phonon number

    Parameters
    ----------
    ion0: ions class object
       the object that represent the system to be simulated
    state_type: type of state used
        0 for density matrix
        1 for ket
    Returns
    -------
    Qutip operator

    '''
    Ns = ion0.df_spin()
    if Ns == 1:
        isket = fock(2,0)
    else:    
        isket = tensor(fock(2,0),fock(2,1)) # ion 1 in excited state
    ini_sdm = isket*isket.dag()
    if ion0.df_phonon()[0] == 1: #only consider one phonon space
        for m in range(ion0.df_phonon()[1][0]):
            if m == 0:
                pho = fock(ion0.pcut[0],0)
            else:
                pho = tensor(pho,phon.fock(ion0.pcut[m],0))
    else:
        for m in range(ion0.df_phonon()[1][0]):
            if m == 0:
                pho1 = fock(ion0.pcut[0][0],0)
            else:
                pho1 = tensor(pho1,fock(ion0.pcut[0][m],0))
        for m in range(ion0.df_phonon()[1][1]):
           if m == 0:
               pho2 = fock(ion0.pcut[1][0],0)
           else:
               pho2 = tensor(pho1,fock(ion0.pcut[1][m],0))  
        pho = tensor(pho1,pho2)       
    dpmat = pho*pho.dag()
    rho0 = tensor(ini_sdm,dpmat)
    if state_type == 0:
        return rho0
    else:
        return tensor(isket,pho)