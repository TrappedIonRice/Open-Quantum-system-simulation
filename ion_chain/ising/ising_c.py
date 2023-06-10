# -*- coding: utf-8 -*-
"""
Compute the complete time-dependent Hamiltonian for the ising coupling system
function: Htot
@author: zhumj
"""
import numpy as np
from qutip import *
import Qsim.ion_chain.ising.ising_ps as iscp
import Qsim.operator.spin as spin
import Qsim.operator.spin_phonon as sp_op
import Qsim.ion_chain.interaction.spin_phonon as Isp
from  Qsim.ion_chain.ion_system import *

def summary():
    print("____________________________________________________________________")
    print("function: H_ord")
    print("Genearte the complete time-dependent Hamiltonian for ising coupling (sx,sy) system with magentic field coupled with sz")
'''
subfunction
'''

'''
function to use
''' 
def H_ord(Bz,ion0,laser0):
    '''
    Genearte the complete time-dependent Hamiltonian for ising coupling (sx,sy) 
    system with magentic field coupled with sz, assuming two symmetric red/blue sidebands
    Parameters
    ----------
    H0 : qutip operator
       time independent part of the Hamiltonian
    ion0: ions class object
        contains all parameters of the ion-chain system
    laser0: laser class object 
        blue sideband laser drive
    Returns
    -------
    Heff : list
        list of Qutip Operator and string expressions for time dependent functions, 
        format required by the Qutip solver
    Hargd : dictionary
      dictionary that records the value of coefficients for time dependent functions
    '''
    Ns = ion0.df_spin
    H0 = tensor(iscp.HBz(ion0,Bz),sp_op.p_I(ion0))
    Heff = [H0]+ Isp.H_td(ion0,laser0,0,1) + Isp.H_td(ion0,laser0,1,1)
    H_arg = Isp.H_td_arg(ion0,laser0)
    return Heff, H_arg
def H_com_multi(ion0,laser_r,laser_b,laser_rc,laser_bc,q):
    '''
    Genearte the time-dependent Hamiltonian for 3-body coupling, assuming all drives
    coupled to the com mode and apply rwa by neglecting all terms faster or at the 
    order of com eigenfrequnecy.
    Parameters
    ----------
    ion0 : ion class object
    laser_r : laser class object
        red sideband laser
    laser_b :  laser class object
        blue sideband laser
    laser_rc : laser class object
       compensation red sideband laser
    laser_bc : laser class object
       compensation blue sideband laser
    q : float
        scale factor for compensation sidebands
    Returns
    -------
    Hlist : list of operators, str
        time dependent Hamiltonian, input of sesolve
    arg_dic1 : dict
        parameter dict

    '''
    arg_dic1 = {'r':1j*2*np.pi*laser_r.mu,'rc':1j*2*np.pi*laser_rc.mu,
                'b':1j*2*np.pi*laser_b.mu,'bc':1j*2*np.pi*laser_bc.mu,
                'w1' : 1j*fr_conv(ion0.radial_freq[0],'khz'),
                'w2' : 1j*fr_conv(ion0.radial_freq[1],'khz'),
                'w3' : 1j*fr_conv(ion0.radial_freq[2],'khz'),
                'q':q}
    Hlist = []
    for i in laser_r.laser_couple:
        s_down = spin.down(ion0.df_spin,i)
        for m in sp_op.ph_list(ion0):
            m_expr = 'w'+str(m+1)
            coefr = 1j/2 * Isp.g(ion0,laser_r,i,m)
            op_fir =  tensor(s_down,sp_op.p_ladder(ion0,laser_r,m,1));
            #first term 
            expr_r1 = ('exp( 1 * t * '+ m_expr + ' ) *  (' 
                       + 'exp( 1 * t * r) + '
                       + 'sqrt(q) * exp( 1* t * rc))' )
            Hr1 = [coefr * op_fir, expr_r1]
            #second term
            expr_r2 = ('exp( -1 * t * '+ m_expr + ' ) *  (' 
                       + 'exp( -1 * t * r) + '
                       + 'sqrt(q) * exp( -1* t * rc))' )
            Hr2 = [-coefr * op_fir.dag(), expr_r2]
            '''
            print('__________________________')
            print('index a: i,m',[i,m])
            print('frequency 1:', np.abs(ion0.radial_freq[m]*1000+laser_r.mu))
            print('frequency 2:', np.abs(ion0.radial_freq[m]*1000+laser_rc.mu))
            '''
            Hlist = Hlist + [Hr1,Hr2]
            for n in sp_op.ph_list(ion0):
                n_expr = 'w'+str(n+1)
                coefb = -1/4 * Isp.g(ion0,laser_b,i,m)*Isp.g(ion0,laser_b,i,n)/ laser_b.Omega(ion0)
                op_sec =  tensor(s_down,sp_op.p_ladder(ion0,laser_r,m,0)*sp_op.p_ladder(ion0,laser_r,n,0))
                #third term, conjugate of first 
                sum_mn = '(' + m_expr + ' + ' + n_expr + ')'
                expr_b1 = ('exp( -1 * t * '+ sum_mn + ' ) *  (' 
                           + 'exp( 1 * t * b) + '
                           + 'sqrt(q) * exp( 1* t * bc))' )
                H3 = [coefb* op_sec, expr_b1]
                #fourth term, conjugate of second 
                expr_b2 = ('exp( 1 * t * '+ sum_mn + ' ) *  (' 
                           + 'exp( -1 * t * b) + '
                           + 'sqrt(q) * exp( -1* t * bc))' )
                '''
                print('__________________________')
                print('index a: i,m,n',[i,m,n])
                print('frequency 3:', np.abs((-ion0.radial_freq[m]-ion0.radial_freq[n])*1000+laser_b.mu))
                print('frequency 4:', np.abs((-ion0.radial_freq[m]-ion0.radial_freq[n])*1000+laser_bc.mu))
                '''
                H4 = [coefb* op_sec.dag(), expr_b2]
                Hlist = Hlist + [H3,H4]
    return Hlist, arg_dic1