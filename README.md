# Open-Quantum-system-simulation
package: ion_chain

dependence pakcage: Qutip

Installation: download clone the repository to C:\users\"your username"\.ipython\Qsim
The module should be import using import Qsim... as .

Functions:

ion_system: define the class ions that can be used to store all physical parameters of ion-chain system with one laser drive, and compute basic physical quantities of the trapped ion system. 

sub-package opeartor:

operator\spin: generate spin operators acting on the N ion spin space

operator\phonon: generate phonon operators acting on the N ion phonon space

sub-package interaction:

interaction\spin_phonon: Compute ion-laser interaction Hamiltonian in resonant/ordinary interaction frame

interaction\pure_spin: Construct spin interaction Hamiltonian for single-site electron transfer systems and double site-excitation transfer systems

sub-package ising:

ising\ising_ps: Generate the ion-laser Hamiltonian under pure spin approximation

ising\ising_c: Compute the complete time-dependent ion-laser Hamiltonian for the ising coupling system

sub-package transfer:

transfer\elec_transfer: Construct Hamiltonian in reasonate rotating frame for the 2 ion open qunatum system used to simulation electron transfer between acceptor and donor state in one site

transfer\exci_transfer: Construct Hamiltonian in reasonate rotating frame for the 3 ion open qunatum system used to simulate excitation transition between 2 sites

transfer\exci_operators: Construct quantum operators used in excitation transfer systems 

transfer\exci_operators: Compute the complete time-dependent Hamiltonian with anharmonic terms for  3 ion open qunatum system, laser field is only coupled to the ion on the side

transfer\multicore: Functions for multi-core parallel computation using package multiprocess

sub-package transfer: eigendiagram

eigendiagram\exci_diagram: Plot energydiagram under a semi-classical approximation for 3 ion open qunatum system used tosimulate excitation transition between 2 sites

Example codes

Example 0: gives the basic commands to use the modules 

Example 1: Computes the time evolution of the ising coulping ion system with a complete Hamiltonian and compare the result with a pure spin interaction approximation

Example 2a: Compute the time evolution of a 2 ion system contructed to simulate electron transfer. Reproduce curve C in Fig3B of Schlawin et. al.'s PRXQuantum Paper.

Example 2b: Compute the time evolution of a 2 ion system contructed to simulate electron transfer with 1 mode or 2 modes. Compare the result using 1 mode (PRXpapaer), 2 mode (special interaction frame), time dependent H (ordinary frame) and test the validity of changing interaction frames

Example 3a: Compute the time evolution of a 3 ion system contructed to simulate excitation transfer between 2 sites using Hamiltonian in reasonant interaction frame

Example 3b:Compute the time evolution of a 3 ion system contructed to simulate excitation transfer between 2 sites in reasonant interaction frame and using time-dependent Hamiltonian in ordinary interaction frame. Verify the results are the same.

Example 3c: plot the energy diagram for excitation transfer Hamiltonian under semi-classical approximation

Example 4a: Generate list of site energy differences for multicore parallel computation

Example 4b: Simulate the time evolution of excitation transfer at different Delta E with multi cores parallel coumputation

Example 4c: Simulate the time evolution of excitation transfer at different dissipation gammawith multi cores parallel coumputation

Example 5a: Compute basic quantities of the anharmonic coupling terms 

Example 5b: Compute ordinary frame phonon evolution for a anharmonic simulator of 3 ions, considering both tilt and rock mode for 2 vibrational directions

Example 5c: Compute ordinary frame phonon evolution for a anharmonic simulator of 3 ions, only considering coupled modes