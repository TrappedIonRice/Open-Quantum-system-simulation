# -*- coding: utf-8 -*-
"""
Construct quantum operators used in excitation transfer systems 

@author: zhumj
"""
import numpy as np
import Qsim.operator.spin as spin
import Qsim.operator.phonon as phon
from  Qsim.ion_chain.ion_system import *
def summary():
    print("____________________________________________________________________")
    print("function: ph_list")
    print("Generate a list of phonon index used in computing laser-ion coupling")
    print("____________________________________________________________________")
    print("function: p_num")
    print("find the number of phonon spaces coupled to the laser")
    print("____________________________________________________________________")
    print("function: p_zero")
    print("Construct the zero operator on phonon spacer")
    print("____________________________________________________________________")
    print("function: p_I")
    print("Construct the identity operator on phonon space")
    print("____________________________________________________________________")
    print("function: p_ladder")
    print("Construct the ladder operator on phonon space")
    print("____________________________________________________________________")
    print("function: rho_thermal")
    print("Construct initial density matrix according to a thermal distribution")
    print("____________________________________________________________________")
    print("function: ini_state")
    print("Construct initial ket/density matrix that has integer phonon number")
    print("____________________________________________________________________")
    print("function: c_op")
    print("Construct the collapse operator for the transfer systems")
    print("____________________________________________________________________")
    print("function: spin_measure")
    print("Construct operators to measure spin evolution for excitation transfer")
    print("____________________________________________________________________")
    print("function: phonon_measure")
    print("Construct operators to measure phonon evolution for excitation transfer")
def ph_list(ion0):
    '''
    Generate a list of phonon index used in computing laser-ion coupling

    Parameters
    ----------
    ion0 : ion class object
    Returns
    -------
    list of int
    '''
    if ion0.df_phonon() [0]== 1: #only consider one phonon space
        mlist = ion0.active_phonon[0]
    else:   #two  phonon spaces
       mlist = ion0.active_phonon[ion0.df_laser]
    return mlist    

def pnum(ion0,df=None):
    '''
    find the number of phonon spaces coupled to the laser
    
    Parameters
    ----------
    ion0 : ion class object
    df : int, default as none
        vibrational degree of freedom that couples to the laser, 0: axial, 1: radial
        Specified if doing computations with a different coupling direction from the direction
        initialized in ion class object

    Returns
    -------
    int, number of phonon spaces coupled to the laser

    '''
    if df == None:
        df_couple = ion0.df_laser
    else:
        df_couple = df
    if ion0.df_phonon() [0]== 1: #only consider one phonon space
        dim = ion0.df_phonon()[1][0]
    else:   #two  phonon spaces
        dim = ion0.df_phonon()[1][df_couple]
    return dim    

def p_zero(ion0):
    '''
    construct the zero operator on phonon space
    Parameters
    ----------
    ion0 : ion class object

    Returns
    -------
    Qutip Operator

    '''
    Np = pnum(ion0)
    pcut =ion0.pcut
    if ion0.df_phonon()[0] == 1: #only consider one phonon space
        pzero = phon.zero_op(pcut[0],Np)
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

    Returns
    -------
    Qutip Operator

    '''
    Np = pnum(ion0)
    pcut =ion0.pcut
    if ion0.df_phonon()[0] == 1: #only consider one phonon space
        pI = phon.pI(pcut[0],Np)
    else:     #two  phonon spaces
        pI = tensor(phon.pI(pcut[0],ion0.df_phonon()[1][0]),
                    phon.pI(pcut[1],ion0.df_phonon()[1][1]))
    return pI

def p_ladder(ion0,mindex,atype,df=None):
    '''
    construct the ladder operator on phonon space
    Parameters
    ----------
    ion0 : ion class object
    mindex: int  
        index of phonon space where the ladder operator is acting on    
    atype: int 
        type of phonon operator, 0 for down, 1 for up
    df : int, default as none
         vibrational degree of freedom that couples to the laser, 0: axial, 1: radial
         Specified if doing computations with a different coupling direction from the direction
         initialized in ion class object    
    Returns
    -------
    Qutip Operator

    ''' 
    
    if df == None:
        df_couple = ion0.df_laser #use default
        Np = pnum(ion0)
    else:
        df_couple = df #specify the coupling coefficeint
        Np = pnum(ion0,df=df_couple)
    pcut = ion0.pcut
    if ion0.df_phonon()[0] == 1: #only consider one phonon space
        if atype == 0:
            opa = phon.down(mindex,pcut[0],Np)
        else:
            opa = phon.up(mindex,pcut[0],Np)
    else:     #two  phonon spaces
        if atype == 0:
            opa = phon.down(mindex,pcut[df_couple],Np)
        else:
            opa = phon.up(mindex,pcut[df_couple],Np)
        #construct in order axial, transverse
        if  df_couple ==0:
            opa = tensor(opa,phon.pI(pcut[1],ion0.df_phonon()[1][1]))
        else:
            opa = tensor(phon.pI(pcut[0],ion0.df_phonon()[1][0]),opa)
    return opa    

def rho_thermal(ion0):
    '''
    Construct initial density matrix according to a thermal distribution
    Parameters
    ----------
    ion0: ions class object
       the object that represent the system to be simulated
       
    Returns
    -------
    Qutip operator

    '''
    Ns = ion0.df_spin()
    wmlist0 = [ion0.Axialfreq()*ion0.fz,ion0.Transfreq()*ion0.fz]
    if Ns == 1:
        isket = fock(2,0)
    else:    
        isket = tensor(fock(2,0),fock(2,1)) # ion 1 in excited state
    ini_sdm = isket*isket.dag()
    if ion0.df_phonon()[0] == 1: #only consider one phonon space
        for mindex in range(ion0.df_phonon()[1][0]):
            m = ph_list(ion0)[mindex]
            wm = ion0.wmlist()[m]
            if mindex == 0:
                pho = phon.inip_thermal(ion0.pcut[0][0],fr_conv(wm,'khz'),ion0.Etot)
            else:
                pho = tensor(pho,phon.inip_thermal(ion0.pcut[0][mindex],fr_conv(wm,'khz'),ion0.Etot))
    else:
        for mindex in range(ion0.df_phonon()[1][0]):
            m = ph_list(ion0)[mindex]
            wm = wmlist0[0][m]
            if mindex == 0:
                pho1 = phon.inip_thermal(ion0.pcut[0][0],fr_conv(wm,'khz'),ion0.Etot)
            else:
                pho1 = tensor(pho1,phon.inip_thermal(ion0.pcut[0][mindex],fr_conv(wm,'khz'),ion0.Etot))
        for mindex in range(ion0.df_phonon()[1][1]):
            m = ph_list(ion0)[mindex]
            wm = wmlist0[1][m]
            if m == 0:
                pho2 = phon.inip_thermal(ion0.pcut[1][0],fr_conv(wm,'khz'),ion0.Etot)
            else:
                pho2 = tensor(pho2,phon.inip_thermal(ion0.pcut[1][mindex],fr_conv(wm,'khz'),ion0.Etot))        
    #dmat = fock(ion0.pcut,0)*fock(ion0.pcut,0).dag()
    #pho0 = tensor(dmat,dmat,dmat)
        pho = tensor(pho1,pho2)
    rho0 = tensor(ini_sdm,pho)
    return rho0    

def ini_state(ion0,s_num,p_num,state_type):
    '''
    Construct initial ket/density matrix that has integer phonon number

    Parameters
    ----------
    ion0: ions class object
       the object that represent the system to be simulated
    s_num: list of int
        specify initial spin state, 0 for up, 1 of down, default as 0
    p_num: list of list of int 
        specified phonon number for the state
    state_type: type of state to be generated 
        0 for density matrix
        1 for ket
    Returns
    -------
    Qutip operator

    '''
    Ns = ion0.df_spin()
    if Ns == 1:
        isket = fock(2,s_num[0])
    else:    
        isket = tensor(fock(2,s_num[0]),fock(2,s_num[1])) # ion 1 in excited state
    ini_sdm = isket*isket.dag()
    if ion0.df_phonon()[0] == 1: #only consider one phonon space
        for m in range(ion0.df_phonon()[1][0]):
            if m == 0:
                pho = fock(ion0.pcut[0][0],p_num[0][m])
            else:
                pho = tensor(pho,phon.fock(ion0.pcut[0][m],p_num[0][m]))
    else:
        for m in range(ion0.df_phonon()[1][0]):
            if m == 0:
                pho1 = fock(ion0.pcut[0][0],p_num[0][m])
            else:
                pho1 = tensor(pho1,fock(ion0.pcut[0][m],p_num[0][m]))
        for m in range(ion0.df_phonon()[1][1]):
            if m == 0:
                pho2 = fock(ion0.pcut[1][0],p_num[1][m])
            else:
                pho2 = tensor(pho2,fock(ion0.pcut[1][m],p_num[1][m]))  
        pho = tensor(pho1,pho2)       
    dpmat = pho*pho.dag()
    rho0 = tensor(ini_sdm,dpmat)
    if state_type == 0:
        return rho0
    else:
        return tensor(isket,pho)

def c_op(ion0,normalized=True):
    '''
    Construct the collapse operator for the transfer systems
    Parameters
    ----------
    ion0 : ion class object
    normalized: bool
        if normalized, all cooling coefficient will be multiplied by
        corresponding Eigenmode matrix element
    Returns
    -------
    List of Qutip operator

    '''
    clist = []
    mindex = 0
    if ion0.df_laser == 0:
        emat = ion0.Axialmode()
    else:
        emat = ion0.Transmode()
    for m in ph_list(ion0):
        cm = tensor(spin.sI(ion0.df_spin()), p_ladder(ion0,mindex,0))
        if normalized:
            coeff = np.abs(emat[m,ion0.coolant[0]])*np.sqrt(fr_conv(ion0.gamma[m],'hz'))
        else:
            coeff = np.sqrt(fr_conv(ion0.gamma[m],'hz'))
        clist.append(coeff*np.sqrt(1+ion0.n_bar()[m])*cm)
        clist.append(coeff* np.sqrt(ion0.n_bar()[m])*cm.dag())
        mindex = mindex + 1                                            
    return clist

def spin_measure(ion0,index):
    '''
    Generate operators to measure spin evolution for excitation transfer systems

    Parameters
    ----------
    ion0 : ion class object
    index : list of int
        specify the spin state to be projected, 0 for spin up, 1 for spin down
        [0,1] means up, down state
    Returns
    -------
    s_op : Qutip operator

    '''
    if ion0.df_spin() == 1:
        s_ket = fock(2,index)
    else:
        s_ket = tensor(fock(2,index[0]),fock(2,index[1]))
    s_op = tensor(s_ket*s_ket.dag(), p_I(ion0))
    return s_op

def phonon_measure(ion0,mindex,df=None):
    '''
    Generate operators to measure phonon evolution for excitation transfer systems
    Parameters
    ----------
    ion0 : ion class object
    mindex: int  
        index of phonon space where the ladder operator is acting on    
    df : int, default as none
         vibrational degree of freedom that couples to the laser, 0: axial, 1: radial
         Specified if doing computations with a different coupling direction from the direction
         initialized in ion class object    
    Returns
    -------
    Qutip operator.

    '''
    if df == None:
        p_op = p_ladder(ion0,mindex,1)*p_ladder(ion0,mindex,0)
    else:     
        p_op = p_ladder(ion0,mindex,1,df)*p_ladder(ion0,mindex,0,df)
    p_op = tensor(spin.sI(ion0.df_spin()),p_op)
    return p_op    