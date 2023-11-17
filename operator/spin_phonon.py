# -*- coding: utf-8 -*-
"""
Construct quantum operators used in excitation transfer systems 
@author: zhumj
"""
import numpy as np
import Qsim.operator.spin as spin
import Qsim.operator.phonon as phon
from  Qsim.ion_chain.ion_system import *
import sigfig
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
    print("function: spin_measure")
    print("Construct operators to measure spin evolution for excitation transfer")
    print("____________________________________________________________________")
    print("function: phonon_measure")
    print("Construct operators to measure phonon evolution for excitation transfer")
def ph_list(ion0,df=1):
    '''
    Generate a list of phonon index used in computing laser-ion coupling
    Parameters
    ----------
    ion0 : ions class object
    df : int, 
        vibrational degree of freedom that couples to the laser, 0: axial, 1: radial
        Specified if doing computations with a different coupling direction from the direction
        initialized in ion class object, in case of two radial spaces, df is used as an index
        to distinguish them.
    Returns
    -------
    list of int
    '''
    if ion0.df_phonon() [0]== 1: #only consider one phonon space
        mlist = ion0.active_phonon[0]
    else:   #two  phonon spaces
    #check if two radial df
       if isinstance(ion0,Ions_asy):
           df = df - 1
       mlist = ion0.active_phonon[df]
    return mlist    

def pnum(ion0, df):
    '''
    find the number of phonon spaces coupled to the laser
    
    Parameters
    ----------
    ion0 : ions class object
    df : int, 
        vibrational degree of freedom that couples to the laser, 0: axial, 1: radial
        Specified if doing computations with a different coupling direction from the direction
        initialized in ion class object, in case of two radial spaces, df is used as an index
        to distinguish them.
    Returns
    -------
    int, number of phonon spaces coupled to the laser
    '''
    if ion0.df_phonon() [0]== 1: #only consider one phonon space
        dim = ion0.df_phonon()[1][0]
    else:   #two  phonon spaces
    #check if two radial df
        if isinstance(ion0,Ions_asy):
            df = df - 1    
        dim = ion0.df_phonon()[1][df]
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
    pcut =ion0.pcut
    if ion0.df_phonon()[0] == 1: #only consider one phonon space
        pzero = phon.zero_op(pcut[0],ion0.df_phonon()[1][0])
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
    pcut =ion0.pcut
    if ion0.df_phonon()[0] == 1: #only consider one phonon space
        pI = phon.pI(pcut[0],ion0.df_phonon()[1][0])
    else:     #two  phonon spaces
        pI = tensor(phon.pI(pcut[0],ion0.df_phonon()[1][0]),
                    phon.pI(pcut[1],ion0.df_phonon()[1][1]))
    return pI

def p_ladder(ion0,df=1, mindex=0,atype=0):
    '''
    construct the ladder operator on phonon space
    Parameters
    ----------
    ion0 : ion class object
    laser0 : laser class object
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
    
    Np = pnum(ion0,df)
    pcut = ion0.pcut
    if ion0.df_phonon()[0] == 1: #only consider one phonon space
        if atype == 0:
            opa = phon.down(m = mindex,cutoff = pcut[0], N = Np)
        else:
            opa = phon.up(m = mindex,cutoff = pcut[0], N = Np)
    else:     #two  phonon spaces
    #check if two radial df
        if isinstance(ion0,Ions_asy):
            df = df - 1    
        if atype == 0:
            opa = phon.down(m = mindex, cutoff = pcut[df], N = Np)
        else:
            opa = phon.up(m = mindex, cutoff = pcut[df], N = Np)
        #construct in order axial, transverse
        if  df ==0:
            opa = tensor(opa,phon.pI(pcut[1],ion0.df_phonon()[1][1]))
        else:
            opa = tensor(phon.pI(pcut[0],ion0.df_phonon()[1][0]),opa)
    return opa    

def rho_thermal(ion0, nbar_list=[],s_config=['z0'], ket = False, s_state=None):
    '''
    Construct initial density matrix/ket for pure state according to a thermal distribution
    Parameters
    ----------
    ion0: ions class object
       the object that represent the system to be simulated
    ket: bool, default as false
        if true, output state as ket for a pure superposition of fock states
        if false, output the usual density matrix used for thermal state
    s_config: list of int. used to initialize the system is a pure spin up/down state
        specify initial spin state, 0 for up, 1 of down, default as 0 
    nbar_list: list of list of float
        average phonon number of each phonon space
    s_state: Qutip density matrix
        a density matrix for spin space of the system
    Returns
    -------
    Qutip operator
    '''
    if s_state == None:
        Ns = ion0.df_spin
        isket = spin.spin_state(s_config)
        ini_sdm = isket*isket.dag()
    else:
        ini_sdm = s_state
    if ion0.df_phonon()[0] == 1: #only consider one phonon space
        for mindex in range(ion0.df_phonon()[1][0]):
            nbar = nbar_list[0][mindex]
            if mindex == 0:
                pho = phon.inip_thermal(ion0.pcut[0][0],nbar,ket)
            else:
                pho = tensor(pho,phon.inip_thermal(ion0.pcut[0][mindex],nbar,ket))
    else:
        for mindex in range(ion0.df_phonon()[1][0]):
            nbar = nbar_list[0][mindex]
            if mindex == 0:
                pho1 = phon.inip_thermal(ion0.pcut[0][0],nbar,ket)
            else:
                pho1 = tensor(pho1,phon.inip_thermal(ion0.pcut[0][mindex],nbar,ket))
        for mindex in range(ion0.df_phonon()[1][1]):
            nbar = nbar_list[1][mindex]
            if mindex == 0:
                pho2 = phon.inip_thermal(ion0.pcut[1][0],nbar,ket)
            else:
                pho2 = tensor(pho2,phon.inip_thermal(ion0.pcut[1][mindex],nbar,ket))        
    #dmat = fock(ion0.pcut,0)*fock(ion0.pcut,0).dag()
    #pho0 = tensor(dmat,dmat,dmat)
        pho = tensor(pho1,pho2)
    if ket:
        ket0 = tensor(isket,pho)
        return ket0
    else:
        rho0 = tensor(ini_sdm,pho)
        return rho0    

def ini_state(ion0=None,s_config=['z0'], p_state = [[0]], state_type=0):
    '''
    Construct initial ket/density matrix that has integer phonon number
    Parameters
    ----------
    ion0: ions class object
       the object that represent the system to be simulated
    s_state: list of int
        specify initial spin state, 0 for up, 1 of down
    p_state: list of list of int 
        specified phonon number for the state
    state_type: type of state to be generated 
        0 for density matrix
        1 for ket
    Returns
    -------
    Qutip operator
    '''
    Ns = ion0.df_spin
    isket = spin.spin_state(s_config)
    ini_sdm = isket*isket.dag()
    if ion0.df_phonon()[0] == 1: #only consider one phonon space
        for m in range(ion0.df_phonon()[1][0]):
            if m == 0:
                pho = fock(ion0.pcut[0][0],p_state[0][m])
            else:
                pho = tensor(pho,phon.fock(ion0.pcut[0][m],p_state[0][m]))
    else:
        for m in range(ion0.df_phonon()[1][0]):
            if m == 0:
                pho1 = fock(ion0.pcut[0][0],p_state[0][m])
            else:
                pho1 = tensor(pho1,fock(ion0.pcut[0][m],p_state[0][m]))
        for m in range(ion0.df_phonon()[1][1]):
            if m == 0:
                pho2 = fock(ion0.pcut[1][0],p_state[1][m])
            else:
                pho2 = tensor(pho2,fock(ion0.pcut[1][m],p_state[1][m]))  
        pho = tensor(pho1,pho2)       
    dpmat = pho*pho.dag()
    rho0 = tensor(ini_sdm,dpmat)
    if state_type == 0:
        return rho0
    else:
        return tensor(isket,pho)

def spin_measure(ion0,s_config=['z0'],s_state=None):
    '''
    Generate operators to measure spin evolution for excitation transfer systems
    Parameters
    ----------
    ion0 : ion class object
    index : list of int, used only for tensor product basis states
        specify the spin state to be projected, 0 for spin up, 1 for spin down
        [0,1] means up, down state
    state: Qutip ket
        specify the spin state to be projected
    Returns
    -------
    s_op : Qutip operator
    '''
    if s_state == None:
        s_ket = spin.spin_state(s_config)
        s_mat = s_ket*s_ket.dag()
    else:
        s_mat = s_state * s_state.dag()
    s_op = tensor(s_mat, p_I(ion0))
    return s_op
def site_spin_measure(ion0=None,index=0):
    '''
    Generate operators to measure site spin population for excitation transfer systems
    p = 0.5*(<\sigma_z>)+0.5
    Parameters
    ----------
    ion0 : ion class object
    index : int
        specify the index of spin space to be measured
    Returns
    -------
    s_op : Qutip operator
    '''
    s_op = tensor( 0.5 * (spin.sI(ion0.df_spin) + spin.sz(ion0.df_spin,index)),
                  p_I(ion0))
    return s_op
def phonon_measure(ion0, df=1, mindex=0):
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
    p_op = p_ladder(ion0,df,mindex,1)*p_ladder(ion0,df,mindex,0)
    p_op = tensor(spin.sI(ion0.df_spin),p_op)
    return p_op    

def pstate_measure(ion0, df=1,meas_level=0,mindex=0):
    '''
    measure the population of n=pcut state of a specific phonon space
    in order to check the validity using a finite phonon space
    ion0 : ion class object
    df : int, default as none
         vibrational degree of freedom that couples to the laser, 0: axial, 1: radial
    meas_level: int
        phonon state level to be measured    
    mindex: int  
        index of phonon space to be measured    
    
    Returns
    -------
    Qutip operator.
    '''
    Np = pnum(ion0, df)
    pcut = ion0.pcut
    if ion0.df_phonon()[0] == 1: #only consider one phonon space
        opa = phon.state_measure(pcut[0],Np,meas_level,mindex)
    else:     #two  phonon spaces
    #check if two radial df
        if isinstance(ion0,Ions_asy):
            df = df - 1    
        opa = phon.state_measure(pcut[df],Np,meas_level,mindex)
        #construct in order axial, transverse
        if  df ==0:
            opa = tensor(opa,phon.pI(pcut[1],ion0.df_phonon()[1][1]))
        else:
            opa = tensor(phon.pI(pcut[0],ion0.df_phonon()[1][0]),opa)
    return tensor(spin.sI(ion0.df_spin),opa)   

def phonon_cutoff_error(states, ion0, df=1, mindex=0,plot=False,log_scale=True):
    '''
    Compute the maximum occupation of the highest allowed phonon state
    as the error for choosing a finite phonon cutoff, and plot the phonon distribution in 
    fock basis of the corresponding state. 

    Parameters
    ----------
    states : list of states (result.state)
        state at different times extracted from qutip sesolve/mesolve result
    ion0 : ion class object
    df : int, default as none
         vibrational degree of freedom that couples to the laser, 0: axial, 1: radial x
         2: radial y (only for Ions_asy subclass)
   mindex: int  
       index of phonon space to be measured    
    plot : bool, optional
        If true, plot phonon distribution, The default is False.

    Returns
    -------
    p_max : float
        cutoff error to the second significant digit.

    '''
    if ion0.df_phonon()[0] == 1: #only consider one phonon space
        meas_level = ion0.pcut[0][mindex]-1
        #index for tracing
        tindex = ion0.df_spin + mindex
    else:     #two  phonon spaces
    #check if two radial df
        if isinstance(ion0,Ions_asy):
            pdf = df - 1    
        meas_level = ion0.pcut[pdf][mindex]-1
        if pdf == 0:
            tindex = ion0.df_spin + mindex
        if pdf == 1: 
            tindex = ion0.df_spin + ion0.df_phonon()[1][0] + mindex
    p_high = expect(pstate_measure(ion0, df, meas_level, mindex),states)
    max_index = np.argmax(p_high)
    p_max = sigfig.round(p_high[max_index],2)
    print('Estimated phonon cutoff error: ', p_max)
    if plot:
        plt.figure()
        pstate = states[max_index].ptrace([tindex])
        plot_fock_distribution(pstate)
        if log_scale:
            plt.yscale('log')
            plt.ylim(np.min(pstate.diag())/10,10*np.max(pstate.diag()))
        plt.show()
    return p_max
    
    
    
    
        
     
    
    
    