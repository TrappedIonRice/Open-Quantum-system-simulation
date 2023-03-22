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
from scipy.optimize import fsolve
from scipy.optimize import minimize
def summary():
    print("____________________________________________________________________")
    print("function: H_ord")
    print("Genearte the time-dependent Hamiltonian for 1 site electron tranfer with anhormonic terms in ordinary interaction frame")

'''
subfunction
'''

def anh_str(dtype,sign):
    '''
    Generate time depedent expression for the anharmonic Hamiltonian
    in resonant/ordinary interaction frame, in form exp(+- d1 * t) exp(+-d2 * t)
    d1 = -j * mu_1, d2 = j(2*mu_2-mu_1) for resonant frame
    in ordinary interaction frame, for simplification, only consider the coupling between
    axial tilt and radial rock mode:
    d1 = -j * \omega_t, d2 = j(2*omega_r-omega_t)
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
        
def ord_ah_arg(ion0):
    '''
    Generate an argument dictonary which maps parameters in time-dependent 
    expressions for anharmonic term in resonant frame to their actual values
    for simplification, only consider the coupling between
    axial tilt and radial rock mode
    Parameters
    ----------
    ion0: ion class object
    Returns
    -------
    dictionary for argument mapping

    '''
    omega_t = 2000*np.pi * ion0.fz * ion0.Axialfreq()[1] #axial tilt freq #akk with unit [2pi kHz]
    omega_r = 2000*np.pi * ion0.fz * ion0.Transfreq()[2] #radial rock freq
    return {'d1':-1j* omega_t,'d2':1j*(2*omega_r-omega_t)}
def res_ah_arg(mu1,mu2):
    '''
    Generate an argument dictonary which maps parameters in time-dependent 
    expressions for anharmonic term in ordinary frame to their actual values

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
def max_axial_eq(fz,*para):
    '''
    construct the equation used for solving maximum axial confining frequnecy
    allowed for a stable 1D chain 

    Parameters
    ----------
    fz : TYPE
        DESCRIPTION.
    *para : tuple
        parameters to construct the equation
        [fx,N]
        fx: float
            radial confining frequency
        N: int
            number of ions in the system
    Returns
    -------
    LHS of the equation 

    '''
    fx0,N = para
    ion_temp = ions()
    ion_temp.N = N
    ion_temp.fx = fx0; ion_temp.fz = fz
    mu = (ion_temp.Axialfreq()[N-1])**2#maximum axial eigenfrequency
    lhs = ion_temp.alpha() - 2/(mu-1)
    del ion_temp
    return lhs
def res_freq_eq(fz,*para):
    '''
    construct the equation used for solving resonant axial confining freq
    for the setup of 1 drive which satisfies \omega_r-\omega

    Parameters
    ----------
    fz : float
        radial trapping frequency, variable to be solved [Mhz]
    *para : tuple
        parameters to construct the equation
        [fx,w]
    Returns
    -------
    LHS of the equation 

    '''
    fx0, w0 = para
    return np.sqrt(3)*fz - 2*np.sqrt(fx0**2 - 12*fz**2/5) + w0/1000
def res_freq_eq2(fz,*para):
    '''
    construct the equation used for solving axial confining freq that statsfies resonant condition
    \omega_m + \omega_n = \nu_p + \omega 

    Parameters
    ----------
    fz : float
        radial trapping frequency, variable to be solved [Mhz]
    *para : tuple
        parameters to construct the equation
        [fx,w, N, m ,n , p]
        fx is the radial confining freq in MHz
        w is the off resonance factor in kHz
        N is the number of ions
        m is the index for first radial mode
        n is the index for second radial mode
        p is the index for axial mode
    Returns
    -------
    LHS of the equation 

    '''
    fx0, w0, N, m, n, p = para
    #create a ion class object
    ion_temp = ions()
    ion_temp.N = N
    ion_temp.fx = fx0; ion_temp.fz = fz
    #compute axial frequency at index p
    afreq = fz*ion_temp.Axialfreq()[p]
    #compute the raidal frequency using another method to avoid 
    #compute radial frequency at index m, n
    rfreq_m = fz*ion_temp.Transfreq()[m]
    rfreq_n = fz*ion_temp.Transfreq()[n]
    del ion_temp
    return np.abs(afreq - (rfreq_m+rfreq_n) + w0/1000)
def fz_cons(fz,*para):
    '''
    construct constraint conditions for a 1D stable equilibrium 

    Parameters
    ----------
    fz : float
        radial trapping frequency, variable to be solved [Mhz]
    *para : tuple
        parameters to construct the equation    
    Returns
    -------
    None.

    '''
    (fmax,) = para
    farray = np.zeros(2)
    farray[0] = fmax-fz
    farray[1] = fz
    return  farray
'''
function to use
''' 
def max_axial(fx,N):
    '''
    maximum axial confining frequnecy allowed for a stable 1D chain 
    Parameters
    ----------
    fx : float
        axial trapping frequency [MHz]
    N: int
        number of ions

    Returns
    -------
    float, radial trapping frequency fz [MHz]

    '''
    fz_i = fx/2
    para0 = (fx,N)
    return fsolve(max_axial_eq, fz_i, args=para0)[0]
def res_freq(fx,w):
    '''
    solve for resonant axial confining freq for the setup of 1 drive which satisfies
    \omega_r-\omega

    Parameters
    ----------
    fx : float
        axial trapping frequency [MHz]
    w: float
        detuning from radial rock mode, [kHz]    

    Returns
    -------
    float, radial trapping frequency fz [MHz]

    '''
    #guess of solution
    fz_i = (20/63)**0.5 * fx
    para0 = (fx,w)
    return fsolve(res_freq_eq, fz_i, args=para0)[0]
def res_freq2(fx,w,N,mode_index,fz_i=1):
    '''
    solve for resonant axial confining freq that statsfies resonant condition
    \omega_m + \omega_n = \nu_p + \omega 

    Parameters
    ----------
    fx : float
        axial trapping frequency [MHz]
    w: float
        detuning from radial rock mode, [kHz]    
    N: int
        number of ions
    m: 
        index for first radial mode
    n:
        index for second radial mode
    p:
        index for axial mode
    f_zi: float
        guess of solution, default is 1 MHz    
    Returns
    -------
    float, radial trapping frequency fz [MHz]

    ''' 
    [m,n,p] = mode_index
    para0 = (fx,w,N,m,n,p)
    #print(para1)
    ub_fz =  max_axial(fx, N)-1e-2
    cons = ({'type': 'ineq', 'fun': lambda fz:ub_fz-fz},
        {'type': 'ineq', 'fun': lambda fz: fz})
    return minimize(res_freq_eq2, fz_i, method = 'trust-constr', args=para0, constraints = cons)['x'][0]
def H_ord1(Omegax, Omegaz, ion0, ah_term=False,ah_op=0,full_ah=False):
    '''
    Compute the complete time-dependent Hamiltonian for the 3 ion open qunatum system
    used to simulate electron of a single site with anharmonicity in ordinary interaction frame with a single
    laser drive, 
    Input:
    ----------
    Omegax : float 
        coupling coefficient between the doner and acceptor state [kHz]
    Omegaz : float
        energy difference between the doner and acceptor state  [kHz]
    ion0: ions class object
        the object that represent the system to be simulated
    
    an_term: bool
        if anharmonic terms will be included
    an_op: Qutip operator/ list of qutip operators
        anharmonic coupling operator with coupling strength
        if using time-dependent anharmonic term, implement the off-res and near-res
        terms with a list 
    full_ah: bool, default as false
        if true, use the full time-dependent form of anharmonic terms       
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
    Heff = Isp.H_td(ion0,0) + Isp.H_td(ion0,1)
    H_arg = Isp.H_td_arg(ion0)
    if not full_ah: #resonant anharmonic term
        if ah_term:
            term_a = ah_op + ah_op.dag()
        else:    
            term_a = 0
        H0 = H_s+term_a
        Heff = ([H0]  + Heff) 
    else: #time-dependnent anharmonic term    
        ahlist = []
        for tindex in range(2):
            ahlist.append([ah_op[tindex], anh_str(tindex,1)])
            ahlist.append([ah_op[tindex].dag(),anh_str(tindex,0)])
        Heff = ([H_s]+ Heff + ahlist)
        H_arg = H_arg | ord_ah_arg(ion0) 
    return Heff, H_arg   
def H_ord2(Omegax, Omegaz, ion1, ion2, ah_term=False,ah_op=0,full_ah = False):
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
    an_op: Qutip operator/ list of qutip operators
        anharmonic coupling operator with coupling strength
        if using time-dependent anharmonic term, implement the off-res and near-res
        terms with a list 
    full_ah: bool, default as false
        if true, use the full time-dependent form of anharmonic terms     
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
    H_arg = Isp.H_td_arg(ion1,'1')|Isp.H_td_arg(ion2,'2')   
    Heff = (Isp.H_td(ion1,0,0,'1') + Isp.H_td(ion1,1,0,'1')
            + Isp.H_td(ion2,0,0,'2') + Isp.H_td(ion2,1,0,'2'))
    if not full_ah: #resonant anharmonic term
        if ah_term:
            term_a = ah_op + ah_op.dag()
        else:    
            term_a = 0
        H0 = H_s+term_a
        Heff = ([H0]  + Heff) 
    else: #time-dependnent anharmonic term    
        ahlist = []
        for tindex in range(2):
            ahlist.append([ah_op[tindex], anh_str(tindex,1)])
            ahlist.append([ah_op[tindex].dag(),anh_str(tindex,0)])
        Heff = ([H_s]+ Heff + ahlist)
        H_arg = H_arg | ord_ah_arg(ion1) 
    return Heff, H_arg
def H_res1(Omegax, Omegaz, ion0, ion_a, ah_term=False,ah_op=0):
    '''
    Compute the complete time-dependent Hamiltonia for the 3 ion open qunatum system
    used to simulate electron of a single site with anharmonicity in resonat interaction frame with 1 laser drive
    following the derivation in PRA 032617
    Input:
    ----------
    Omegax : float 
        coupling coefficient between the doner and acceptor state [kHz]
    Omegaz : float
        energy difference between the doner and acceptor state  [kHz]
    ion0: ions class object
        used to construct H for laser dirve 1, describes the coupling to radial modes
    ion_a: ions class object
        describes the additional harmonic term for axial mode
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
    '''
    H_s =  Is.single_site(Omegax, Omegaz, ion0)
    H_a = Isp.H_harmonic(ion_a)
    H0 = H_s + Isp.H_res(ion0) - H_a 
    if ah_term:
        term_a = ah_op + ah_op.dag()
    else:    
        term_a = 0
    Heff = H0+term_a
    return Heff
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
            H_arg = res_ah_arg(ion2.mu(),ion1.mu())
            ahlist = []
            for tindex in range(2):
                ahlist.append([ah_op[tindex], anh_str(tindex,1)])
                ahlist.append([ah_op[tindex].dag(),anh_str(tindex,0)])
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