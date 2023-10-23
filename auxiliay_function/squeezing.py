# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 23:27:39 2023
This module is used for simuation of spin-squeezing generation. The purpose is to
directly set the parameters without using the ion class.
@author: zhumj
"""
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import Qsim.operator.spin as spin
import Qsim.operator.phonon as phon
import sigfig
from scipy.optimize import minimize
def unit_vec_conv(phi,theta):
    '''
    generate cartesian coordiants of a  vector on 
    unit sphere given two angular parameters
    '''
    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)
    return np.array([x,y,z])
def polar_conv(n_vec):
    '''
    Convert a unit vector in cartesian coorinate in to
    polar coordinate theta, phi

    Parameters
    ----------
    n_vec : np array
        unit vector

    Returns
    -------
    spherical coordinates 

    '''
    phi = np.arccos(n_vec[2])
    if (n_vec[0] == 0 and n_vec[1] == 0):
        theta = 0
    else:
        xy_proj = np.sqrt(n_vec[0]**2+n_vec[1]**2)
        if n_vec[1]>0:
            theta = np.arccos(n_vec[0]/xy_proj)
        else:
            theta = 2*np.pi - np.arccos(n_vec[0]/xy_proj)
    return [phi,theta]
def rotation_trans(n_vec):
    '''
    Conostruct the transformation matrix that converts the coordinate in rotated 
    system (z for MSD) to the original coordinate system (z for Sz)
    

    Parameters
    ----------
    n_vec: np array of 3d vector
        new z axis (MSD vec) expressed in original coordinate
    Returns
    -------
    3*3 np matrix

    '''
    #convert to polar coordinate
    [phi,theta] = polar_conv(n_vec)
    Rmat = np.zeros((3,3))
    Rmat[0] = [np.cos(theta)*np.cos(phi), -np.sin(theta), np.cos(theta)*np.sin(phi)]
    Rmat[1] = [np.sin(theta)*np.cos(phi), np.cos(theta), np.sin(theta)*np.sin(phi)]
    Rmat[2] = [-np.sin(phi), 0, np.cos(phi)]
    return Rmat
def orthognal_unit_vec(vec,theta):
    '''
    generate unit vectors orthogonal to a given MSD vec

    Parameters
    ----------
    vec : np array of float
        MSD unit vector
    theta: float
        second angular coordinate
    Returns
    -------
    unit vector, np array 

    '''
    new_coord = [np.cos(theta),np.sin(theta),0]
    return np.dot(rotation_trans(vec),new_coord)
def var_orthogonal(state,N_spin, theta):
    '''
    For a given state and a polar coorinate theta, compute the 
    variance of orthogonal spin operator

    Parameters
    ----------
    state : Qutip operator
        state of N spin system
    theta : float
        polar coordinate

    Returns
    -------
    float, spin squeezing parameter epsilon_H

    '''
    n_vec0 = spin.MSD(state,N_spin)
    n_vec = orthognal_unit_vec(n_vec0 ,theta)
    Jn_per = spin.Jn_operator(n_vec, N_spin)
    return variance(Jn_per,state)
def min_var_eq(theta,*para):
    '''
    function to be minized for finding minimum orthogonal variance
    '''
    theta = theta[0]
    state= para[0]; N_spin = para[1]; 
    return var_orthogonal(state,N_spin, theta)
def min_var_para(state,N_spin):
    '''
    Find minimum spin squeezing in N spin system

    Parameters
    ----------
    state : Qutip operator
        state of N spin system
    N_spin : int
        number of spin space in the system

    Returns
    -------
    float

    '''
    para0 = (state,N_spin)
    #estimate the angle for minimum variance
    est_list = np.linspace(0, 2*np.pi,100)
    varlist = []
    for phi in est_list:
        varlist.append(var_orthogonal(state, N_spin, phi))
    start = np.min(np.array(varlist))        
    return minimize(min_var_eq, start, args=para0 )
def sq_para(var,state,N_spin):
    n_vec0 = spin.MSD(state,N_spin)
    Jn = spin.Jn_operator(n_vec0, N_spin)
    return 2*var/ (expect(Jn,state))**2
#analytic formula for minimum vairance
def min_var_analytic(state,N_spin):
    #compute vector n1, n2
    n_vec = spin.MSD(state,N_spin)
    [phi,theta] = polar_conv(n_vec)
    n1 = np.array([-np.sin(theta),np.cos(theta),0])
    n2 = np.array([np.cos(phi)*np.cos(theta),
                   np.cos(phi)*np.sin(theta),
                   -np.sin(phi)])
    J1 = spin.Jn_operator(n1,N_spin)
    J2 = spin.Jn_operator(n2,N_spin)
    term1 = expect(J1* J1+ J2 * J2,state)
    term2 = expect(J1*J1 - J2*J2, state)
    term3 = 0.5*expect(J1*J2 + J2*J1, state)
    result = 0.5*(term1 - 
                  np.sqrt(term2**2 + 4*term3**2))
    return result
def optimal_squeezing_angle(state,N_spin,plot=False):
    #compute optimal squeezing angle
    n_vec = spin.MSD(state,N_spin)
    [phi,theta] = polar_conv(n_vec)
    n1 = np.array([-np.sin(theta),np.cos(theta),0])
    n2 = np.array([np.cos(phi)*np.cos(theta),
                   np.cos(phi)*np.sin(theta),
                   -np.sin(phi)])
    print(n1,n2)
    J1 = spin.Jn_operator(n1,N_spin)
    J2 = spin.Jn_operator(n2,N_spin)
    term1 = expect(J1* J1+ J2 * J2,state)
    A = expect(J1*J1 - J2*J2, state)
    B = expect(J1*J2 + J2*J1, state)
    #print(A,B)
    if B <= 0:
        phi = 0.5*np.arccos(-A/np.sqrt(A**2+B**2))
    else:
        phi = np.pi - 0.5*np.arccos(-A/np.sqrt(A**2+B**2))
    if plot:
        #construct the vector on Bloch sphere
        opt_vec =  n1*np.cos(phi) + n2*np.sin(phi)
        bs_plot = qutip.Bloch()
        bs_plot.make_sphere()
        bs_plot.add_vectors(n_vec)
        bs_plot.add_vectors(opt_vec)
        bs_plot.show()
    return phi
def H_res(delta=0,E=0,V=0,Omega=0,cutoff=2, alpha = 1,op_type = 'x'):
    '''
    construct the Hamiltonian for the model, consider 
    rocking mode only, the second ion is used as coolant

    Parameters
    ----------
    delta : float, optional
        harmonic energy level. The default is 0.
    E : float, optional
        sigma_y coupling. The default is 0.
    V : float, optional
        sigma_x coupling. The default is 0.
    Omega : float, optional
        spin phonon coupling. The default is 0.
    cutoff : int, optional
        phonon space cutoff. The default is 2.

    Returns
    -------
    None.

    '''
    #operators
    a_up = tensor(spin.sI(2),phon.up(m=0,cutoff=[cutoff],N=1))
    a_down = tensor(spin.sI(2),phon.down(m=0,cutoff=[cutoff],N=1))
    sigma_x1 = tensor(spin.sx(N=2,i=0),phon.pI([cutoff],1)) 
    sigma_x3 = tensor(spin.sx(N=2,i=1),phon.pI([cutoff],1)) 
    sigma_y1 = tensor(spin.sy(N=2,i=0),phon.pI([cutoff],1)) 
    sigma_y3 = tensor(spin.sy(N=2,i=1),phon.pI([cutoff],1)) 
    sigma_z1 = tensor(spin.sz(N=2,i=0),phon.pI([cutoff],1)) 
    sigma_z3 = tensor(spin.sz(N=2,i=1),phon.pI([cutoff],1)) 
    # harmonic term
    Hh = delta * a_up * a_down
    #spin term
    Hs = E * (sigma_y1+sigma_y3) + V * (sigma_x1+sigma_x3)
    #spin phonon coupling
    if op_type == 'x':
        spin_op = sigma_x1 + alpha*sigma_x3
        print('spin phonon operator: sigma_x')
    elif op_type == 'xy+':
        spin_op = (sigma_x1+sigma_y1) + alpha*(sigma_x3+sigma_y3)
        print('spin phonon operator: sigma_x + sigma_y')
    elif op_type == 'xy-':
        spin_op = (sigma_x1-sigma_y1) + alpha*(sigma_x3-sigma_y3)
        print('spin phonon operator: sigma_x - sigma_y')
    elif op_type == 'z':
        spin_op = sigma_z1 + alpha*sigma_z3
        Hs = E * (sigma_z1+sigma_z3) + V * (sigma_x1+sigma_x3)
        print('spin phonon operator: sigma_z')
    Hsp = Omega * (a_up+a_down) * spin_op
    H0 = 2*np.pi* (Hh+Hs+Hsp)
    return H0
def cooling(gamma=0, nbar=0, cutoff=2):
    clist = []
    cm = tensor(spin.sI(2),phon.down(m=0,cutoff=[cutoff],N=1))
    coeff = np.sqrt(2*np.pi*gamma)
    clist.append(coeff*np.sqrt(1+nbar)*cm)
    clist.append(coeff* np.sqrt(nbar)*cm.dag())                                         
    return clist  

def init_state(spin_state, p_num, cutoff):
    ket0 = tensor(spin_state,fock(cutoff,p_num))
    rho = ket0*ket0.dag()
    return rho
def phonon_measure(cutoff):
    op = tensor(spin.sI(2),
                phon.up(m=0,cutoff=[cutoff],N=1)*phon.down(m=0,cutoff=[cutoff],N=1))
    return op
def phonon_cutoff_error(states,cutoff):
    '''
    Compute the maximum occupation of the highest allowed phonon state
    as the error for choosing a finite phonon cutoff, and plot the phonon distribution in 
    fock basis of the corresponding state. 

    Parameters
    ----------
    states : list of states (result.state)
        state at different times extracted from qutip sesolve/mesolve result
    Returns
    -------
    p_max : float
        cutoff error to the second significant digit.

    '''
    opa = tensor(spin.sI(2),phon.state_measure([cutoff],1,cutoff-1,0))
    p_high = expect(opa,states)
    max_index = np.argmax(p_high)
    p_max = sigfig.round(p_high[max_index],2)
    print('Estimated phonon cutoff error: ', p_max)
    plt.figure()
    pstate = states[max_index].ptrace([2])
    plot_fock_distribution(pstate)
    plt.yscale('log')
    plt.ylim(np.min(pstate.diag())/10,10*np.max(pstate.diag()))
    return p_max