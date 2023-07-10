# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 17:02:00 2023

@author: zhumj
"""

import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import Qsim.operator.spin as spin
import Qsim.operator.phonon as phon
import Qsim.ion_chain.ising.ising_ps as iscp
import Qsim.ion_chain.ising.ising_c as iscc
import Qsim.operator.spin_phonon as sp_op
import Qsim.ion_chain.interaction.spin_phonon as Isp
from  Qsim.ion_chain.ion_system import *
from scipy import signal
import copy
#%%
ion_sys = Ions_asy()
ion_sys.freq_offset = 500 
ion_sys.update_trap()
ion_sys.list_para()