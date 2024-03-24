# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 13:33:03 2022

@author: zhumj
functions for multi-core parallel computation using package multiprocess
"""

import Qsim.ion_chain.transfer.exci_transfer as extrans
import numpy as np
from qutip import *
import multiprocessing as mp
from tqdm import tqdm
import datetime
import pandas as pd
from operator import itemgetter 
#from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_DOWN

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

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
def generate_task(core_num, var_list, para_list=()):
    '''
    generate an np array that has higher resolution at expected position of the peaks 
    Parameters

    Parameters
    ----------
    core_num : int
        number of cores used for parallel computation
    var_list : list
        list of variables for each simualtion 
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
    vlength = len(var_list)
    '''
    if vlength < core_num:
        core_num = vlength
    task_length = Decimal(str(vlength/core_num)).quantize(Decimal('1.'), rounding=ROUND_HALF_DOWN)
    print(int(task_length))
    splitter = np.arange(0, vlength, int(task_length))[1:]
    print(splitter)
    if np.size(splitter) < core_num:
        splitter = np.append(splitter,vlength)
    elif np.size(splitter) > core_num:
        splitter = np.delete(splitter,vlength)
    else:
        splitter[np.size(splitter)-1] = vlength
    inlist = np.split(iarray,splitter)    
    '''
    #generate a index array 
    iarray = np.arange(0,vlength)
    inlist = list(split(list(iarray), core_num))
    #generate task dictionary
    tdict = {}
    for i in range(core_num):
        new_vlist = itemgetter(*list(inlist[i]))(var_list)
        print(isinstance(new_vlist, tuple))
        if not isinstance(new_vlist, tuple):
            new_vlist= (new_vlist,)
            
        tdict[str(i)] = new_vlist  + para_list
    #print('task dictionary', tdict)
    return tdict
  
def ME_multi_H(task,Hlist,sim_para):
    '''
    solve time evolution for a single energy splitting
    Parameters
    ----------
    task : string 
        task name
    Hlist : list of Qutip operators
        Hamiltonians to be simulated
    sim_para : dict 
        A dictionary that takes the following format:
           {
            'rho' : , (initial density matrix)
            't_array':, (time array for sampling dynamics)
            'elist' : , (list of observables to be evaluated)
            'clist' :  (list of collapse operators to construct Lindbladian operator)
            }
    Returns
    -------
        task: task name
        sresult: a list of arrays that represents updown state population
        evolution at different deltaE
    '''
    #read parameters:
    rho_s = sim_para['rho'] ; t_array_s = sim_para['t_array'] ; 
    clist_s = sim_para['clist'] ; elist_s = sim_para['elist'] ; 
    
    sresult = []
    for H in Hlist:
        result = mesolve(H,rho_s,t_array_s,clist_s,elist_s,progress_bar=False,options=Options(nsteps=100000))
        rhoee = result.expect[0]
        sresult.append(rhoee)
    return {task:sresult}       

def multi_H_parallel(task_func,sim_para,Hlist,n_cpu):
    '''
    Simulate electron transfer given a list of Hamiltonian using parallel computing
    
    Parameters
    ----------
    task_func: python function
        function to be called for simualtion
    sim_para : dict 
        A dictionary that takes the following format:
            {
             'rho' : , (initial density matrix)
             't_array':, (time array for sampling dynamics)
             'elist' : , (list of observables to be evaluated)
             'clist' :  (list of collapse operators to construct Lindbladian operator)
             }
    Hlist : list of qutip operators
        the list of H to be simulated 
    n_cpu: int
        number of cpu to be used for simluation 
    Returns
    -------
    p_result, list of all simulation results

    '''
    tdict = generate_task(n_cpu,Hlist)
    #print('task dictionary', tdict) 
    #if __name__ == '__main__':
    print('start parallel computing')
    print('number of cores used:',n_cpu,'/',mp.cpu_count())
    start_t = datetime.datetime.now() #record starting time
    pool = mp.Pool(n_cpu)
    results = [pool.apply_async(task_func, args=(ntask, H_array,sim_para)) 
               for ntask, H_array in tdict.items()]
    pool.close()
    result_list_tqdm = [] #generate progress bar
    for result in tqdm(results):
        result_list_tqdm.append(result.get()) 
    end_t = datetime.datetime.now()
    elapsed_sec = (end_t - start_t).total_seconds()
    print("time consumed " + "{:.2f}".format(elapsed_sec) + "s")    
    sevl = [] #combine all the results
    for i in range(n_cpu):
        sevl = sevl + result_list_tqdm[i][str(i)]
    print('________________________________________________________________')   
    print('all computation completed')    
    return sevl