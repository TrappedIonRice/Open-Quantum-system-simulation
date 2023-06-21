# -*- coding: utf-8 -*-
"""
Plot Wigner function and generate Gif
"""
from  Qsim.ion_chain.ion_system import *
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sigfig
import imageio
import matplotlib as mpl
from datetime import datetime
import os


#phase space coordinates for ploting Wigner function
xvec = np.linspace(-5,5,200)
X, Y = np.meshgrid(xvec, xvec) 


#function for plotting Wigner function at a specfic frame 
def generate_wplot(t_index, sim_time, result, proj, p_index, 
                   state_type=0,save=False, img_index = 0):
    '''
    plot wigner function at a certain frame index

    Parameters
    ----------
    t_index : int
        index of frame to be plotted 
    sim_time : np array
        time array used for qutip simulation 
    result : array
        array of result states, output of qutip solver
    proj : qutip operator
        projection operator to be applied on the state
    p_index: int
        index of phonon space to be traced out, value would depend on the 
        structure of tensor product space 
    state_type: int, default as 0
        type of result state, 0 for ket, 1 for density matrix
    save: bool, default as False
        if True, save the Wigner function plot
    img_index: int default as 0
        index of saved image, used for generating gif
    Returns
    -------
    list of center of distribution, (maximum amplitude), coordinate pair

    '''
    new_st = result.states[t_index]
    if state_type == 0:
        new_st = new_st*new_st.dag()*proj
    else:
        new_st = new_st*proj
    rho_p = new_st.ptrace(p_index) #trace out phonon dm0
    wdist = wigner(rho_p, xvec, xvec)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    wlim = abs(wdist).max()
    #wmap = wigner_cmap(wdist)
    cf = ax.contourf(xvec, xvec, wdist, 100,norm=mpl.colors.Normalize(-wlim, wlim),cmap=cm.RdBu)
    plt.axhline(0, color='black', linewidth=1.5)  # plot solid horizontal line at y=0
    plt.axvline(0, color='black', linewidth=1.5)
    [max_x,max_y] = np.unravel_index(wdist.argmax(), wdist.shape)
    #plt.scatter(xvec[max_x], xvec[max_y], color='red', marker='.')
    plt.scatter(X[max_x,max_y] , Y[max_x,max_y] , color='red', marker='.')
    ax.set_xlabel(r'$x$', fontsize=14)
    ax.set_ylabel(r'$p$', fontsize=14)
    new_time = sigfig.round(sim_time[t_index],2)
    ax.set_title('t= '+str(new_time)+' [ms]',fontsize = 14)
    ax.tick_params(axis='both', labelsize=13)
    ax.grid()
    if save:
        fig.savefig('frame'+str(img_index)+'.png',dpi=100)
    plt.close()
    return [X[max_x,max_y] , Y[max_x,max_y]]
#functions for generating time evolution gif 
def frame_time(n_frames,sim_time):
    '''
    Generate frame index for time evolution gif, the frames are sampled
    uniformly from the given time array

    Parameters
    ----------
    n_frames : int
        number of frames to be plotted
    sim_time : np array
        time array used for qutip simulation 

    Returns
    -------
    frame_index : np array
        array of python index (size n_frames), each element correponds to the index of a
        frame in sim_time.

    '''
    tstep = int(np.round(len(sim_time)/n_frames))
    frame_index = np.arange(0,len(sim_time),tstep)
    return frame_index 
#functions to use
def wigner_evol_frames(n_frames,sim_time,result, proj, p_index, 
                   state_type=0,folder_name = None):
    '''
    Generate all frame image for time evolution gif, the frames are sampled
    uniformly from given time array

    Parameters
    ----------
    n_frames : int
        number of frames to be plotted
    sim_time : np array
        time array used for qutip simulation 
    result : array
        array of result states, output of qutip solver
    proj : qutip operator
        projection operator to be applied on the state
    p_index: int
        index of phonon space to be traced out, value would depend on the 
        structure of tensor product space 
    state_type: int, default as 0
        type of result state, 0 for ket, 1 for density matrix
    folder_name : str, 
        name of the folder for storage.
        If this folder exists, the frame images will be stored there,
        otherwise the folder will be created under the current working directory.
        The default is None.

    Returns
    -------
    None.

    '''
    #dt_string =datetime.now().strftime("%m_%d_%H_%M_%S")
    #progress bar
    bar_size = n_frames/10;
    last_report = 0;
    bar = 10;
    wplot_times = frame_time(n_frames,sim_time)
    print('start frame generation')
    for i in range(n_frames):
        if i - last_report > bar_size:
            print('percent of tasks completed: '+str(bar)+'%')
            last_report = i;
            bar = bar + 10;
        generate_wplot(wplot_times[i], sim_time, result, proj, p_index, 
                           state_type=0,save=True, img_index = i)
    print('All tasks finished.')
def wiger_evol_gif(n_frames,gif_name,frame_duration=0.5,remove_frame = False):
    '''
    Generate gif for visiualizing the time evolution of wigner function, make 
    sure this function is called in the correct working directory in which all
    frame images are stored

    Parameters
    ----------
    n_frames : int
        
    gif_name : str
        name of output gif
    frame_duration : float, optional
        frame duration in [s]. The default is 0.5.
    remove_frame : bool, optional
        If true, remove frame. The default is False.

    Returns
    -------
    None.

    '''
    all_frames = []
    for i in range(n_frames):
        filename = 'frame'+str(i)+'.png'
        image = imageio.imread(filename)
        all_frames.append(image)
        if remove_frame:
            os.remove(filename)
    kargs = { 'duration': frame_duration}
    imageio.mimsave(gif_name, all_frames, 'GIF', **kargs)
    print(gif_name+' successfully generated.')
    print('Stored at', os.getcwd())