#!/usr/bin/env python3
from Parsing_Tools import *
from Classes import *
import os
import argparse
import logging
import openfermion
import scipy
from methods import *
from timeit import default_timer as timer
import hat

#Parse command line arguments, particularly an input file
start = timer()
parser = argparse.ArgumentParser()
parser.add_argument("input", type=str)
parser.add_argument("output", nargs='?', default='app.log')

args = parser.parse_args()
logging.basicConfig(filename=args.output, filemode='w', format='%(message)s')
logging.getLogger().setLevel(logging.INFO)


hat.hat()
print('*' * 39)
print('MaGE - Mayhall and Grimsley Eigentools')
print('    H. R. Grimsley, N. J. Mayhall')
print('*' * 39)
#Parse the input file to form a molecule object
print('Obtaining molecular data...')
molecule = Get_Molecule(args.input)
logging.info('Geometry: '+str(molecule.geometry))
logging.info('Basis: '+str(molecule.basis))
logging.info('Multiplicity: '+str(molecule.multiplicity))
logging.info('Spin-Orbitals: '+str(molecule.n_orbitals*2))
logging.info('Qubits: '+str(molecule.n_qubits))
logging.info('HF = '+str(molecule.hf_energy))
logging.info('CCSD = '+str(molecule.ccsd_energy))
logging.info('CISD = '+str(molecule.cisd_energy))
logging.info('FCI = '+str(molecule.fci_energy))

#Obtain some useful global data
HF_ket = scipy.sparse.csc_matrix(openfermion.jw_configuration_state(list(range(0,molecule.n_electrons)), molecule.n_qubits)).transpose()
print('Constructing operator bank...')
ops = Operator_Bank(molecule, **Get_Op_Kwargs(args.input))
print(str(len(ops.Full_Ops))+' operations!')
#Run optimization procedure
outcome = Optimize(molecule, ops, logging, **Get_Method_Kwargs(args.input))
end = timer()

#Log results
logging.info('Correlation energy = '+str(molecule.fci_energy-molecule.hf_energy))
logging.info('Optimized energy = '+str(outcome.fun))
logging.info('Error = '+str(outcome.fun-molecule.fci_energy))
logging.info('Iterations = '+str(outcome.nit))
logging.info('Total time (s) = '+str(end-start))
logging.info(str(len(outcome.x))+' parameters.')
print('See output in '+str(args.output)+'!')

