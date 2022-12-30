# -*- coding: utf-8 -*-
"""
This example intends to show how to import modules and read the documentation of functions in this package
Please run one line each time in the console
"""
#%%
#import the spin subpackage from ion_chain.operator and call it "spin" for futher usage
import Qsim.operator.spin as spin
#%%
#print the discription of the package and list the name of all functions
print(spin.__doc__)
#%%
#read the summary for all functions in spin
spin.summary()
#%%
#print the documentation for function sx
help(spin.sx)
#%% this part illustrates how to use 'class' 
#import all functions of ion_system.py from ion_chain.ising 
from  Qsim.ion_chain.ising.ion_system import *
#%% see the documents for ions class
help(ions)
#read document for a specific function in this class
help(ions.n_bar)
#%% construct a ions class object
ions1 = ions()
ions1.list_para() #list defaut value of parameters
#%% change the value of parameter
ions1.pcut = 20
ions1.list_para()
#%% use function of ions class
print(ions1.n_bar()) 
