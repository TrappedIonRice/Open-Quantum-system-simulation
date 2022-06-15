# -*- coding: utf-8 -*-
"""
This example intends to show how to import modules and read the documentation of functions in this package
"""
#%%
#import the spin subpackage from ion_chain.operator and call it "spin" for futher usage
import ion_chain.operator.spin as spin
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
#import the electron transfer subpackage from ion_chain.ising and call it "etrans" for futher usage 
import ion_chain.ising.etransfer as etrans
#%%
#list all functions and classes:
etrans.summary()
#%% see the documents for ions class
help(etrans.ions)
#read document for a specific function in this class
help(etrans.ions.n_bar)
#%% construct a ions class object
ions1 = etrans.ions()
ions1.list_para() #list defaut value of parameters
#%% change the value of parameter
ions1.pcut = 20
ions1.list_para()
#%% use function of ions class
print(ions1.n_bar()) 