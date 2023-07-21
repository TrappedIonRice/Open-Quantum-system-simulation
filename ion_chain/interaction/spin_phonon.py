# -*- coding: utf-8 -*-
"""
Compute ion-laser interaction Hamiltonian in resonant/ordinary interaction frame
@author: zhumj
"""
import numpy as np
from qutip import *
import Qsim.operator.spin as spin
import Qsim.operator.phonon as phon
import Qsim.operator.spin_phonon as sp_op
from  Qsim.ion_chain.ion_system import *
def summary():
    print("____________________________________________________________________")
    print("function: H_td")
    print("Generate time-dependent Hamiltonian for laser-ion interaction in ordinary frame")
    print("____________________________________________________________________")
    print("function: H_td_arg")
    print(" Generate an argument dictonary which maps parameters in time-dependent expressions to their actual values")
    print("____________________________________________________________________")
    print("function: H_res")
    print("Generate time-independent Hamiltonian for laser-ion interaction in resonant frame")
    
'''
subfunctions
'''    
#Used for genearting correct strings, 
#specifies the motional degree freedom of which the laser is coupling to,
df_dic={0:'z',1:'x',2:'y'}

def RWA_filter(Hlist,flist,f_crit):
    '''
    Apply rotating wave approximation by selecting terms in the Hamiltonian
    with frequency smaller than a criterion. 
    Parameters
    ----------
    Hlist : list of time dependent Hamiltonian
        All Hamiltonians being considered
    flist : list of float
        rotating frequency of each Hamiltonian in Hlist
    f_crit : float
        critical frequency for filtering, [2pi Hz]
    Returns
    -------
    list of filtered time-dependent Hamiltonians
    '''
    aop_dic = ['++','+-','-+','--']
    ind_list = []
    new_Hlist = []
    for i in range(len(Hlist)):
        if flist[i]<f_crit:
            new_Hlist.append(Hlist[i])
            ind_list.append(i)
    '''
    if ind_list != []:
        
        print(ind_list)
        for i in ind_list:
            print('frequency kHz')
            print(np.array(flist)[i]/(2*np.pi))
            
            print('exponetial,operator type')
            print(Hlist[i][1])
            print('phonon operator')
            print(aop_dic[i])
   '''
    return new_Hlist

def LD_coef(ion0,laser0,i,m):
    '''
    Compute the laser-ion coupling strength [2pi kHz] between ion i and mode m
    Parameters
    ----------
    ion0: ion class object
    laser0: laser class object
        laser drive coupled to the ion chain system
    i : int
        ion index
    m : int
        eigenmode index
    Returns
    -------
     float unit of 1
    '''
    if laser0.wavevector == 0:
        emat = ion0.axial_mode
    else:   
        emat = ion0.radial_mode
    coeff = laser0.eta(efreq(ion0,laser0)[m])*emat[m,i] 
    return coeff

def g(ion0,laser0,i,m):
    '''
    Compute the laser-ion coupling strength [2pi kHz] between ion i and mode m
    Parameters
    ----------
    ion0: ion class object
    laser0: laser class object
        laser drive coupled to the ion chain system
    i : int
        ion index
    m : int
        eigenmode index
    Returns
    -------
    g : float
        [2pi kHz]
    '''
    return LD_coef(ion0,laser0,i,m)*laser0.Omega(ion0)

def sigma_phi(N,i,phase):
    #print(phase)
    return np.cos(phase)*spin.sx(N,i) + np.sin(phase)*spin.sy(N,i)

def tstring(N=1,atype=0,las_label=''):
    '''
    Generate the list of time depedent expression for the Hamiltonian of two
    symmetric sideband drives
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
        newm = "w" + las_label + str(mi)
        mstring.append(newm)
        if atype == 1:
            fstring.append('cos(t * '+ustr+') * exp(t * ' + newm +")")
        else:
            fstring.append('cos(t * '+ustr+') * exp(-1 * (t * ' + newm +"))")
    return mstring, fstring        

def tstring_general(N=1,df_lab=''):
    '''
    Generate the list of time depedent expression for the Hamiltonian of an arbitrary
    laser drive
    
    Parameters
    ----------
    N : int
        total number of ions in the trapped ion system
    las_label: str, default as ''
        extra label for the laser drive, specify when using more than 1 laser drives
    Returns
    -------
    mstring : list of string
        list of parameters
    '''
    mstring = []
    for mi in range(1,N+1):
        newm = "w"  + str(mi) + df_lab
        mstring.append(newm)
    return mstring   

'''
Functions for two symmetric sidebands
'''

def Him_ord(ion0,laser0, atype=0,i=0,m=0,sindex=0,mindex=0,i_type=0):
    '''
    Compute the i,m th component for time independent part of ion-laser interaction 
    Hamiltonian of two symmetric sideband drives in ordinary frame, 
    which discribes the interaction for ion i, mode m
    Input: 
        ion0: ion class object
        laser: laser class object, represents a set of two symmetric sidebands
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
    p_df = laser0.wavevector
    p_opa = sp_op.p_ladder(ion0,p_df,mindex,atype)
    if i_type == 1:
        s_oper = sigma_phi(ion0.df_spin,sindex,laser0.phase)
    else:    
        s_oper = spin.sz(ion0.df_spin,sindex)
    H = tensor(s_oper,p_opa)
    return g(ion0,laser0,i,m)*H 
def Him_res(ion0, laser0, i=0,m=0,sindex=0,mindex=0):
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
    p_df = laser0.wavevector
    p_opa = sp_op.p_ladder(ion0,p_df,mindex,0) + sp_op.p_ladder(ion0,p_df,mindex,1) 
    H = tensor(spin.sz(ion0.df_spin,sindex),p_opa)
    return 0.5*g(ion0,laser0,i,m)*H 

def H_td(ion0,laser0, atype=0,i_type = 0,las_label=''): 
    '''
    Compute the list of H correponding to time-dependent Hamiltonian for ion-lasesr
    interaction with a pair of symmetric red/blue sidebands drive as a input for qutip solver
    Input: 
        ion0, ion class object
        laser0, laser class object
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
    for m in sp_op.ph_list(ion0):
        sindex = 0 #this index is used for spin operators
        subH = tensor(spin.zero_op(ion0.df_spin),sp_op.p_zero(ion0))
        for i in laser0.laser_couple:
            subH = subH + Him_ord(ion0,laser0,atype,i,m,sindex,mindex,i_type)
            sindex = sindex + 1
        mindex = mindex+1
        Hlist.append([subH,Hexpr[m]]) 
    return Hlist

def H_td_arg(ion0,laser0,las_label=''):    
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
    adic = {"u"+las_label:fr_conv(laser0.mu,'Hz')}
    slist, fs = tstring(ion0.N,0,las_label)
    wlist0 = 1j*efreq(ion0,laser0) * 2000* np.pi #compute eigenfrequency list
    for argi in range(ion0.N):
        adic[slist[argi]] = wlist0[argi]
    return adic 

def H_res(ion0,laser0):
    '''
    Compute the time-independent Hamiltonian e for ion-lasesr
    interaction with a single drive in resonant fram
    Input: 
        ion0, ion class object
    '''
    spterm = tensor(spin.zero_op(ion0.df_spin),sp_op.p_zero(ion0)) #laser-ion interaction term 
    mindex = 0 #this index is used for phonon operators
    for m in sp_op.ph_list(ion0):
        sindex = 0 #this index is used for spin operators
        for i in laser0.laser_couple:
            spterm = spterm + Him_res(ion0,laser0,i,m,sindex,mindex)
            sindex = sindex + 1
        mindex = mindex + 1
    return spterm - H_harmonic(ion0,laser0)

'''
Functions for a single sideband
'''

def Him_td_fir_ord(ion0, laser0, stype = 0,i=0,m=0,sindex=0,mindex = 0, las_label='',
                   rwa = False, arg_dic={}, f_crit=0):
    '''
    Compute  the i,m th component for 1st order time independent part of spin-phonon 
    interaction Hamiltonian of a single laser drive in ordinary frame, 
    discribeing the interaction for ion i, mode m
    Input: 
        ion0: ion class object
        laser0: laser class object
        stype: int
            spin opeartor type, 0 for destroy, 1 for create
        i: int
            ion index 
        m: int
            phonon space index
        sindex: int
            index to construct spin operator
        mindex: int
            index to construct phonon operator
        las_label: str
            string representing laser index, used to generate string expression
            related to mu 
        rwa: bool
            if True, automatically apply rotating wave approximation by neglecting
            terms with frequency larger than rwa criterion. 
        arg_dic: dict
            parameter dict for RWA
        f_crit: float
            rwa criterion, [2pi kHz] neglect all terms with frequency above it           
    Output:
        Hamiltonina H im, Qobj
    '''
    #set coefficient constants according to the coupling degree of freedom
    p_df = laser0.wavevector
    df_lab = df_dic[p_df]
    ustr = 'u'+las_label+df_lab 
    mstr = 'w'+str(m+1)+df_lab
    
    if stype == 1:
        s_op = spin.up(ion0.df_spin,sindex)
        nu_expr = '* exp( -1 * (t * ' + ustr +' ) )'; u_coef = -1
        coef =  1j/2 * g(ion0,laser0,i,m) * np.exp(1j*laser0.phase)
    if stype == 0:
        s_op = spin.down(ion0.df_spin,sindex)
        nu_expr = '* exp( 1 * t * ' + ustr +' )'; u_coef = 1
        coef =  -1j/2* g(ion0,laser0, i,m) * np.exp(-1j*laser0.phase)    
    p_up = sp_op.p_ladder(ion0,p_df,mindex,1); p_down = sp_op.p_ladder(ion0,p_df,mindex,0)
    exp_plus = 'exp(t * ' +  mstr + ' )'; exp_minus = 'exp(-1 * t * ' +  mstr + ' )'
    H1 = [coef*tensor(s_op,p_up),exp_plus+nu_expr]
    H2 = [coef*tensor(s_op,p_down),exp_minus+nu_expr]
    if rwa:
        freq1 = np.abs( arg_dic[mstr] + u_coef * arg_dic[ustr]) 
        freq2 = np.abs(-arg_dic[mstr] + u_coef * arg_dic[ustr])
        filtered_H = RWA_filter([H1,H2],[freq1,freq2], f_crit )#2*np.pi*1e3*ion0.fx)
        if len(filtered_H) != 0:
            
            '''
            print('spin index')
            print(sindex)
            print('spin operator')
            print(['-','+'][stype])
            print('coupling coefficient')
            print(np.abs(coef))
            '''
        return filtered_H 
    else:
        return [H1,H2] 

def Him_td_sec_ord(ion0, laser0, stype = 0,i=0,sindex=0, mindex_list=[0,0,0,0], las_label='',
                   rwa = False, arg_dic = {}, f_crit=0):
    '''
    Compute  the i,m th component for 2nd order time independent part of spin-phonon 
    interaction Hamiltonian of a single laser drive in ordinary frame, 
    discribing the interaction for ion i, mode m
    Input: 
        ion0: ion class object
        laser0: laser class object
        stype: int
            spin opeartor type, 0 for destroy, 1 for create
        i: int
            ion index 
        sindex: int
            index to construct spin operator
        mindex_list: list of int
            list of index for phonon operators  and related paraemeters
            in form: [m_a,mindex_a,m_b,mindex_b]
        las_label: str
            string representing laser index, used to generate string expression
            related to mu 
        rwa: bool
            if True, automatically apply rotating wave approximation by neglecting
            terms with frequency larger than rwa criterion. 
        arg_dic: dict
            parameter dict for RWA
        f_crit: float
            rwa criterion, [2pi kHz] neglect all terms with frequency above it
    Output:
        Hamiltonina H im, Qobj
    '''
    #set coefficient constants according to the coupling degree of freedom
    [m_a,mindex_a,m_b,mindex_b] = mindex_list
    p_df = laser0.wavevector
    df_lab = df_dic[p_df]
    ustr = 'u'+las_label+df_lab
    mstr_a = 'w'+str(m_a+1)+df_lab
    mstr_b = 'w'+str(m_b+1)+df_lab
    if laser0.Omega_eff!=0:
        coef0 = -(1/4) * g(ion0,laser0,i,m_a) * g(ion0,laser0,i,m_b) /laser0.Omega(ion0)
    else:
        coef0 = 0
    if stype == 1:
        s_op = spin.up(ion0.df_spin,sindex)
        nu_expr = '* exp( -1 * (t * ' + ustr +' ) )'; u_coef = -1
        coef =  coef0  * np.exp(1j*laser0.phase)
    if stype == 0:
        s_op = spin.down(ion0.df_spin,sindex)
        nu_expr = '* exp( 1 * t * ' + ustr +' )'; u_coef = 1
        coef =  coef0 * np.exp(-1j*laser0.phase)
    
    p_up_a = sp_op.p_ladder(ion0,p_df,mindex_a,1); 
    p_down_a = sp_op.p_ladder(ion0,p_df,mindex_a,0)
    p_up_b = sp_op.p_ladder(ion0,p_df,mindex_b,1); 
    p_down_b = sp_op.p_ladder(ion0,p_df,mindex_b,0)
    
    exp_plus_a = 'exp(t * ' +  mstr_a + ' )'; 
    exp_minus_a = 'exp(-1 * t * ' +  mstr_a + ' )'
    exp_plus_b = 'exp(t * ' +  mstr_b + ' )'; 
    exp_minus_b = 'exp(-1 * t * ' +  mstr_b + ' )'
    
    H1 = [coef*tensor(s_op,p_up_a*p_up_b),exp_plus_a+' * '+exp_plus_b+nu_expr]
    H2 = [coef*tensor(s_op,p_up_a*p_down_b),exp_plus_a+' * '+exp_minus_b+nu_expr]
    H3 = [coef*tensor(s_op,p_down_a*p_up_b),exp_minus_a+' * '+exp_plus_b+nu_expr]
    H4 = [coef*tensor(s_op,p_down_a*p_down_b),exp_minus_a+' * '+exp_minus_b+nu_expr]
    
    if rwa:
        freq1 = np.abs(arg_dic[mstr_a] +  arg_dic[mstr_b] + u_coef * arg_dic[ustr])
        freq2 = np.abs(arg_dic[mstr_a] -  arg_dic[mstr_b] + u_coef * arg_dic[ustr])
        freq3 = np.abs(-arg_dic[mstr_a] +  arg_dic[mstr_b] + u_coef * arg_dic[ustr])
        freq4 = np.abs(-arg_dic[mstr_a] -  arg_dic[mstr_b] + u_coef * arg_dic[ustr])
        #print(i,sindex,mindex_list)
        #print(np.array([freq1,freq2,freq3,freq4])/(2*np.pi))
        filtered_H = RWA_filter([H1,H2,H3,H4],[freq1,freq2,freq3,freq4],f_crit)# 2*np.pi*1e3*ion0.fx)
        '''
        if len(filtered_H) != 0:
            print('_________________________________________')
            print('index, i , m, n ',[i,m_a,m_b])
            
            print(sindex)
            print('spin operator')
            print(['-','+'][stype])
            print('coupling coefficient')
            print(np.abs(coef))
            '''
        
        return filtered_H
    else:
        return [H1,H2,H3,H4] 
def H_td_multi_drives(ion0, laser_list, second_order = False, rwa = False, arg_dic = {},f_crit=0):
    '''
    constuct time-dependent laser-ion interaction Hamiltonian with multiple laser drives
    under power series expansion up to second order. 
    Parameters
    ----------
    ion0 : ion class object
    laser_list : list of laser object
        all laser sidebands applied
    second_order: bool
        if True, consider second order terms in the power series expansion
    rwa: bool
        if True, automatically apply rotating wave approximation by neglecting
        terms with frequency larger than rwa criterion. 
    arg_dic: dict
        parameter dict for RWA
    f_crit: float
        rwa criterion, [2pi kHz] neglect all terms with frequency above it
    Returns
    -------
    Hlist : list of Qutip operators
        Time-dependent Hamiltonian
    '''
    #compute the mth element by summing over i for Him for destroy operators
    Hlist = []
    las_lab = 1
    for laser in laser_list:
        mindex = 0 #this index is used for phonon operators
        for m in sp_op.ph_list(ion0):
            sindex = 0 #this index is used for spin operators
            for i in laser.laser_couple:
                newH1 = Him_td_fir_ord(ion0,laser, 0,i,m,sindex,mindex,
                                       str(las_lab),rwa,arg_dic,f_crit)
                newH2 = Him_td_fir_ord(ion0,laser, 1,i,m,sindex,mindex,
                                       str(las_lab),rwa,arg_dic,f_crit)
                sindex += 1
                Hlist = Hlist + newH1 + newH2
            mindex += 1 
        las_lab += 1
    #Hlist = []
    #print('constructing second order')
    if second_order:
        las_lab = 1
        for laser in laser_list:
            sindex = 0 #this index is used for spin operators
            for i in laser.laser_couple:
                mindex_a = 0;  #first index for phonon operators
                for m_a in sp_op.ph_list(ion0):
                    mindex_b = 0 #second index for phonon operators
                    for m_b in sp_op.ph_list(ion0):
                        mlist = [m_a,mindex_a,m_b,mindex_b]
                        newH1 = Him_td_sec_ord(ion0, laser, 0, i,sindex,mlist,
                                               str(las_lab),rwa,arg_dic,f_crit)
                        newH2 = Him_td_sec_ord(ion0, laser, 1, i,sindex,mlist,
                                               str(las_lab),rwa,arg_dic,f_crit)
                        Hlist = Hlist + newH1 + newH2
                        mindex_b += 1
                    mindex_a += 1 
                sindex += 1
            las_lab += 1
    return Hlist

def H_td_multi_drives_asy(ion0, laser_list, second_order = False, rwa = False, arg_dic = {},f_crit=0):
    '''
    constuct time-dependent laser-ion interaction Hamiltonian with multiple laser drives
    under power series expansion up to second order. 
    Parameters
    ----------
    ion0 : Ion_asy class object
    laser_list : list of list of laser object
        all laser sidebands applied
    second_order: bool
        if True, consider second order terms in the power series expansion
    rwa: bool
        if True, automatically apply rotating wave approximation by neglecting
        terms with frequency larger than rwa criterion. 
    arg_dic: dict
        parameter dict for RWA
    f_crit: float
        rwa criterion, [2pi kHz] neglect all terms with frequency above it
    Returns
    -------
    Hlist : TYPE
        DESCRIPTION.
    '''
    # x-motional radial mode
    Hlist1 = H_td_multi_drives(ion0, laser_list[0], second_order, rwa, arg_dic, f_crit)
    Hlist2 = H_td_multi_drives(ion0, laser_list[1], second_order, rwa, arg_dic, f_crit)
    return Hlist1 + Hlist2
'''
functions for parametric amplification
'''
def H_m_PA(ion0, m=0, mindex=0 , pa_index=0,df=1):
    '''
    Compute mth term of the Hamiltonian due to parameteric amplification

    Parameters
    ----------
    ion0 : ion class object
    laser0: laser class object, used to specify coupling direction
    m : int
        mode index m
    mindex : int
        index for phonon operator
    df: int
        vibrational degree of freedom where amplification is applied, 0 for axial
        1 for radial, The default is 1.
    pa_index: int
        index of pa component index, used for multiple wave component case
        default as 0
    Returns
    -------
    list of Qutip operators

    '''
    mstr = 'w'+str(m+1)
    fm_str = 'fm'+str(pa_index)
    exp_plus = 'cos(t * ' + fm_str + ') * exp(2 * t * ' +  mstr + ' )'
    exp_minus = 'cos(t * ' + fm_str + ') * exp(-2 * t * ' +  mstr + ' )'
    p_up = sp_op.p_ladder(ion0,df,mindex=mindex,atype=1); 
    p_down = sp_op.p_ladder(ion0,df,mindex=mindex,atype=0)
    coef = ion0.PA_coef(df,m,pa_index)
    #time independent part of H
    cross_term = tensor(spin.sI(ion0.df_spin), p_up*p_down)
    H0 = [coef* (cross_term + cross_term.dag()),'cos(t * '+fm_str+')']
    H1 = [coef * tensor(spin.sI(ion0.df_spin),p_down*p_down), exp_minus]
    H2 = [coef * tensor(spin.sI(ion0.df_spin),p_up*p_up), exp_plus]
    return [H0,H1,H2]
def H_PA_td(ion0, pa_index=0, df=1):
    '''
    compute time depedent Hamiltonian for parametric amplification
    Parameters
    ----------
    ion0 : ion class object
        DESCRIPTION.
    pa_index: int
        index of pa component index, used for multiple wave component case
        default as 0
    df : int, optional
        vibrational degree of freedom where amplification is applied, 0 for axial
        1 for radial, The default is 1.

    Returns
    -------
    Hlist : list
        time depedent Hamiltonian
    padic : dict
        parameter dict

    '''
    fm_str = 'fm'+str(pa_index)
    padic = {fm_str:fr_conv(ion0.f_mod[pa_index],'Hz')}
    Hlist = []
    mindex = 0
    for m in sp_op.ph_list(ion0):
        new_H = H_m_PA(ion0, m, mindex, pa_index, df)
        Hlist += new_H
        mindex +=  1
    return Hlist, padic

def H_harmonic(ion0,laser0):
    '''
    Compute the harmonic part of the spin-phonon interaction Hamiltonian in
    resonant frame
    Input: 
        ion0: ion class object
    Output:
        Qutip operator
    '''
    p_df = laser0.wavevector
    hterm = tensor(spin.zero_op(ion0.df_spin),sp_op.p_zero(ion0)) #compensation for change of interaction frame
    mindex = 0 #this index is used for phonon operators
    dlist = laser0.detuning(ion0)
    for m in sp_op.ph_list(ion0):
        hterm = (hterm + dlist[m]
                 *tensor(spin.sI(ion0.df_spin),
                         sp_op.p_ladder(ion0,p_df,mindex,1)*sp_op.p_ladder(ion0,p_df,mindex,0)))    
        mindex = mindex+1
    return hterm

def H_td_argdic_general(ion0,laser_list,df_lab=''):    
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
    adic = {}
    df_lab = df_dic[laser_list[0].wavevector]
    for i in range(len(laser_list)):
        adic["u"+str(i+1)+df_lab] = 1j*fr_conv(laser_list[i].mu,'Hz')
    slist = tstring_general(ion0.N,df_lab)
    wlist0 = 1j * efreq(ion0,laser_list[0]) * 2000* np.pi #compute eigenfrequency list
    for argi in range(ion0.N):
        adic[slist[argi]] = wlist0[argi]
    return adic 

