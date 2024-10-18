# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 20:53:19 2023
Compute ET rate based on theoretical models
@author: zhumj
"""
from qutip import *
import Qsim.operator.spin as spin
import Qsim.operator.phonon as phon
import numpy as np
import matplotlib.pyplot as plt
def d_red(V,gf,gamma,times):
    fc = -gf*np.exp(-gf**2/2)
    vp = fc*V
    gp = 0.5*(1+gf**2)
    mat = np.array([[0,-2*vp,0],
                    [vp,-gp*gamma,-vp],
                    [0,2*vp,-gamma]])
    #print(mat)
    eva, U = np.linalg.eig(mat)
    
    Uinv = np.linalg.inv(U)
    #print(U@d_mat@np.linalg.inv(U))
    rho0 = np.array([1,0,0])
    rho11 = []
    rho22 = []
    rho33 = []
    for t in times:
        et = np.exp(np.pi*2*eva*t)
        d_mat = np.diag(et)
        rhot = U@d_mat@Uinv@rho0
        rho11.append(rhot[0])
        rho22.append(rhot[2])
        rho33.append(1-rhot[0]-rhot[2])
    return rho11,rho22,rho33
            