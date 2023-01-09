# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 13:33:03 2022

@author: zhumj
functions for multi-core parallel computation using package multiprocess
"""

import Qsim.ion_chain.transfer.exci_transfer as extrans
import numpy as np
from qutip import *

def summary():
    '''
    give a summary of all functions and classes defined in this module
    '''
    print('___________________________________________________________________')
    print('function: generate_flist')
    print('generate an np array of frequencies that has higher resolution at certain values and lower resolution for other values')
    print('___________________________________________________________________')
    print('function:generate_fplist')
    print('generate an np array of specfied  frequencies')
def generate_flist(start,end,peaks,sep1,sep2,r):
    '''
    generate an np array of frequencies that has higher resolution at certain values
    and lower resolution for other values
    (for instance, array of site energy difference that has higher resolution
    at expected resonances) 
    Parameters
    ----------
    start : float
        left bound of energy interval 
    end : float
        right bound of energy interval 
    peaks : array of peak posiiton
        expected position of the peaks
    sep1 : float
        resolution in ordinary interval
    sep2 : float
        resolution near peak
    r: float
        radias of high resolution region
    Returns
    -------
    Earray
    np array that has higher resolution at expected position of the peaks, input E2list

    '''
    farray0 = np.arange(start,end,sep1)
    ifarray = np.copy(farray0)
    farray = np.array([])    
    #combine 2 arrays
    for i in range(np.size(peaks)):
        if not (peaks[i] in farray0):
            ifarray = np.append(ifarray,peaks[i])
    ifarray = np.sort(ifarray)  
    #print(iEarray)      
    for i in range(np.size(ifarray)):
        ele = ifarray[i]
        if ele in peaks:
            if ele == 0:
                high_e = np.arange(0,ele+r+sep2,sep2) 
            else:    
                high_e = np.arange(ele-r,ele+r+sep2,sep2)    
            farray =  np.concatenate((farray,high_e))
        else:    
            farray = np.concatenate((farray,np.array([ele])))
    return np.unique(farray)    
def generate_fplist(peaks,sep,r):
    '''
    generate an np array of specfied  frequencies
    ----------  
    peaks : array of peak posiiton
        expected position of the peaks
    sep : float
        step near peak
    r: float
        radias of high resolution region
    Returns
    -------
    Earray
    np array that has higher resolution at expected position of the peaks, input E2list

    '''
    farray = np.array([])    
    #combine 2 arrays
    for ele in peaks:
        nflist = np.arange(ele-r,ele+r+sep,sep)
        farray =  np.concatenate((farray,nflist))
    return np.unique(farray)  
def generate_task(core_num,flist, para_list=[]):
    '''
    generate an np array that has higher resolution at expected position of the peaks 
    Parameters

    Parameters
    ----------
    core_num : int
        number of cores used for parallel computation
    flist : np array
        array of a frequency variable, used as the axis for spiltting task array
    ion0: ion class object    
    para_list : np array
        list of parameters to charaterize H, specification is not required for simple cases
        (only varing E or gamma)
        for the complete format: para,ion0, [rho0,elist,tlist] 
        para =[J23,E3,V]
        rho0: initial state of the system    
        elist: operator list to compute expectation value
        tlist: time intervals to compute evolution [times0,times1]
    Returns
    -------
    tdict: dictionary
         {'task index':plist}
         plist is in form [Earray,parray,ions,]
    

    '''
    length = np.size(flist) // core_num
    splitter = np.arange(0,np.size(flist),length)[1:]
    if np.size(flist) % core_num == 0:
        splitter = np.append(splitter,np.size(flist))
    else:
        splitter[np.size(splitter)-1] = np.size(flist)
    inlist = np.split(flist,splitter)
    #generate task dictionary
    tdict = {}
    for i in range(core_num):
        tdict[str(i)] = [inlist[i]]  + para_list
    #print('task dictionary', tdict)
    return tdict
   
def spin_evol(task,parray):
    '''
    solve time evolution for a single energy splitting
    Parameters
    ----------
    task : string 
        task name
    Elist : np array
        input site energy to be computed for the task

    Returns
    -------
        task: task name
        sresult: a list of arrays that represents updown state population
        evolution at different deltaE
    '''
    #read parameters:
    Earray = parray[0]; Hpara = parray[1]
    ion_sys = parray[2]; rho0 = parray[3]
    elist = parray[4]; times0 = parray[5][0]; times1 = parray[5][1]
    J23 = Hpara[0]; E3 = Hpara[1]; V =  Hpara[2]
    
    sresult = []
    for E2 in Earray:
        H0, clist1 = extrans.Htot(J23,E2,E3,V,ion_sys,0)
        #print("solving time evolution for interval 1")
        result0 = mesolve(H0,rho0,times0,clist1,[],progress_bar=True,options=Options(nsteps=10000)) 
        #print("solving time evolution for interval 2")
        result1 = mesolve(H0,result0.states[-1],times1,clist1,elist,progress_bar=True,options=Options(nsteps=100000))
        pplot1 =  np.append(expect(elist[0],result0.states),result1.expect[0])
        sresult.append(pplot1)
    return {task:sresult}       

