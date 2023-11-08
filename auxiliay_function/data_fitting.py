# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:08:08 2023
Functions for curve fitting
@author: zhumj
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy import integrate

'''
fitting electron transfer data
'''
def et_decay(t,A,k,B):
    '''
    exponential decay
    
    '''
    return A*np.exp(-k*t) + B


def norm_et_decay(t, A, k, B):
    '''
    exponential decay normalized to 1

    '''
    return  (1-B)*np.exp(-k * t) + B
def et_osc_decay(t, A,k, B,q, w,phi,C):
    '''
    oscillating exponential decay
    '''
    result =  (A*np.exp(-k*t) +
               B*np.exp(-q*t)*np.cos(w*t+phi) + C)
    return result
'''
functions to use
'''

def et_decay_integrate(tdata,pdata):
    '''
    Extract decay rate by performing a numeric integration

    Parameters
    ----------
    tdata : np array
        normalized time
    pdata : np array
        excited state population

    Returns
    -------
    transfer rate

    '''
    I1 = integrate.simpson(pdata, tdata)
    I2 = integrate.simpson(tdata*pdata, tdata)
    return I1/I2

def fit_et_decay(tdata,pdata,fit_interval='all',plot=False,
                 all_parameter=False):
    '''
    Fit electron transfer excited state population with a exponential 
    decay in form A*np.exp(-k*t) + B 
    Parameters
    ----------
    tdata : np array
        normalized time
    pdata : np array
        excited state population
    fit_interval : string/list, optional
         Interval for data fitting, 
         If 'all',
         all input data points will be used for
         fitting.
         To specify the interval, input a list of 
         two array index as start and end points.
         EX: for an time array of 1000 element,
         fit_interval = [100,200] means points with index 100~200
         will be used for fitting.
         The default is 'all'.
    plot : bool, optional
         If true, plot the fitted curve on the raw data
    all_parameter : bool, optional
        if true, return all parameters fitted for the model
        if false, return the normalized decay rate only
    Returns
    -------
    float, arrays of float
        normalized decay rate/ all parameters of the model 

    '''
    if fit_interval == 'all':
        start = 0; end = np.size(tdata)-1
    else:
        start = fit_interval[0]; end = fit_interval[1]
    ftdata = tdata[start:end]; fpdata = pdata[start:end]
    coef, pcov=curve_fit(norm_et_decay,ftdata,fpdata,p0=[1,0,0],
                         bounds=([0,0,0], [1.5,100,1]),maxfev=5000)
    if plot:
        plt.figure()
        #plot raw data
        plt.plot(tdata,pdata,label='data')
        #plot fit
        plt.plot(tdata, norm_et_decay(tdata, *coef), 'r--',
         label='fit: A=%5.3f, k=%5.3f, B=%5.3f' % tuple(coef))
        plt.ylabel(r'$p_\uparrow$',fontsize = 13)      
        plt.xticks(fontsize = 13)  
        plt.yticks(fontsize = 13)  
        plt.xlabel(r'$\omega_0t/(2\pi)$',fontsize = 13)
        plt.grid()    
        plt.legend()
        plt.show()
    if all_parameter:
        return coef, pcov
    else:
        return coef[1]
    
def fit_et_osc_decay(tdata,pdata,fit_interval='all',plot=False,
                 all_parameter=False,
                 p0=[1,0, 0,0, 1,0,0],
                 bounds=([0,0 ,0,0, 0,-1*np.pi,-1], 
                         [2,100, 2,100, 10,np.pi,1])):
    '''
    Fit electron transfer excited state population with a superposition of 
    exponential decay and an oscillating exponential decay 
    in form Aexp(-kt) + Bexp(-qt)np.cos(wt+phi) + C
    Parameters
    ----------
    tdata : np array
        normalized time
    pdata : np array
        excited state population
    fit_interval : string/list, optional
         Interval for data fitting, 
         If 'all',
         all input data points will be used for
         fitting.
         To specify the interval, input a list of 
         two array index as start and end points.
         EX: for an time array of 1000 element,
         fit_interval = [100,200] means points with index 100~200
         will be used for fitting.
         The default is 'all'.
    plot : bool, optional
         If true, plot the fitted curve on the raw data
    all_parameter : bool, optional
        if true, return all parameters fitted for the model
        if false, return the normalized decay rate only
    p0: list
        initial guess of parameters, in order [A,k, B,q, w,phi,C]
    bounds: tuple of list
        (lower, upper) bounds of the parameters,in order [A,k, B,q, w,phi,C]
    Returns
    -------
    float, arrays of float
        normalized decay rate/ all parameters of the model 

    '''
    if fit_interval == 'all':
        start = 0; end = np.size(tdata)-1
    else:
        start = fit_interval[0]; end = fit_interval[1]
    ftdata = tdata[start:end]; fpdata = pdata[start:end]
    coef, pcov=curve_fit(et_osc_decay,ftdata,fpdata,p0=p0,
                         bounds=bounds,maxfev=5000)
    if plot:
        plt.figure()
        #plot raw data
        plt.plot(tdata,pdata,label='data')
        #plot fit
        plt.plot(tdata, et_osc_decay(tdata, *coef), 'r--', label = 'fit')
         #label=r'fit: A=%5.3f,k=%5.3f, $\omega$=%5.3f, $\phi$=%5.3f, B=%5.3f'  
         #% tuple(coef))
        plt.ylabel(r'$p_\uparrow$',fontsize = 13)      
        plt.xticks(fontsize = 13)  
        plt.yticks(fontsize = 13)  
        plt.xlabel(r'$\omega_0t/(2\pi)$',fontsize = 13)
        plt.grid()    
        plt.legend()
        plt.show()
    if all_parameter:
        return coef, pcov
    else:
        return coef[[1,3]]