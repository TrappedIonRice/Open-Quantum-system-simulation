# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 16:54:10 2023
Compute the ising coupling matrix under a pure spin approximation
@author: zhumj
"""
#%%
from qutip import *
import Qsim.ion_chain.ising.ising_ps as iscp
import Qsim.ion_chain.ising.ising_c as iscc
from  Qsim.ion_chain.ion_system import *
#%%
'''
set parameters of the system
'''    
ion_sys = ions(trap_config={'N': 2, 'fx': 5, 'fz': 0.2}, )
laser1 = Laser()
laser1.mu = ion_sys.fx*1000 + 20
#%%
#ion_sys.N = 4
#ion_sys.laser_couple = [0,1,2,3]
#ion_sys.N = 2
Jmat = iscp.Jt(ion_sys,laser1)
print(Jmat)
iscp.plotj(Jmat)
