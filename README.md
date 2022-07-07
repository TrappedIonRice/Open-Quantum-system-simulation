# Open-Quantum-system-simulation
package: ion_chain

Installation: copy the file ion_chain to C:\users\"your username"\.ipython

Functions:

sub-package opeartor:

operator\spin: generate spin operators acting on the N ion spin space

operator\phonon: generate phonon operators acting on the N ion phonon space

sub-package ising:

ising\ion_system: define the class ions that can be used to store all physical parameters of ion-chain system and environment 

ising\ising_ps: Compute basic physical quantities of ising coupling system and generate the 
Hamiltonian under pure spin approximation

ising\ising_c:Compute the complete Hamiltonian for the ising coupling system

ising\ising_ce:Compute the complete Hamiltonian for the 2 ion open qunatum system used to simulation electron transition between acceptor and donor state

ising\ising_cex: Compute the complete Hamiltonian for the 3 ion open qunatum system used to simulate excitation transition between 2 molecules

sub-package transfer:

transfer\elec_transfer: Construct Hamiltonian in reasonate rotating frame for the 2 ion open qunatum system used to simulation electron transition between acceptor and donor state

transfer\exci_transfer: Construct Hamiltonian in reasonate rotating frame for the 3 ion open qunatum system used to simulate excitation transition between 2 molecules

Example 0 gives the basic commands to use the modules 
