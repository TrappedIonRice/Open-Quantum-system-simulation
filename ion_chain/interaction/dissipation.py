# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 18:26:30 2023
Construct the collapse operator for simulating dissipations in open quantum systems
@author: zhumj
"""
import numpy as np
from qutip import *
import Qsim.operator.spin as spin
import Qsim.operator.spin_phonon as sp_op
from  Qsim.ion_chain.ion_system import *
def cooling(ion0, nbar_list=[1],normalized=True):
    '''
    Construct the collapse operator for the transfer systems
    Parameters
    ----------
    ion0 : ion class object
    nbar_list: list of float
        average phonon number of each phonon space
    normalized: bool
        if normalized, all cooling coefficient will be multiplied by
        corresponding Eigenmode matrix element
    Returns
    -------
    List of Qutip operators
    '''
    clist = []
    mindex = 0
    if ion0.df_cooling == 0:
        emat = ion0.axial_mode
    else:
        emat = ion0.radial_mode
    for m in sp_op.ph_list(ion0,ion0.df_cooling):
        nbar = nbar_list[m]
        cm = tensor(spin.sI(ion0.df_spin), sp_op.p_ladder(ion0,ion0.df_cooling,mindex,0))
        if normalized:
            coeff = np.abs(emat[m,ion0.coolant[0]])*np.sqrt(fr_conv(ion0.gamma[m],'Hz'))
        else:
            coeff = np.sqrt(fr_conv(ion0.gamma[m],'Hz'))
        clist.append(coeff*np.sqrt(1+nbar)*cm)
        clist.append(coeff* np.sqrt(nbar)*cm.dag())
        mindex = mindex + 1                                            
    return clist
def heating(ion0, hr_list, df=1):
    '''
    Construct the collapse operator for simulating in trapped ion system
    ----------
    ion0 : ion class object
    hr_list: list of float
        list of effective heating rate, [kHz]
    df: int
        specify the motional degree of freedom on which the heating is acted on
    Returns
    -------
    List of Qutip operators
    '''
    clist = []
    mindex = 0
    for m in sp_op.ph_list(ion0,df):
        cm = tensor(spin.sI(ion0.df_spin), sp_op.p_ladder(ion0,df,mindex,0))
        coeff = np.sqrt(fr_conv(hr_list[m],'Hz'))
        clist.append(coeff*cm)
        clist.append(coeff*cm.dag())
        mindex = mindex + 1                                            
    return clist