#!/usr/bin/env python3
from Parsing_Tools import *
from Classes import *
import os
import argparse
import logging
import openfermion
import scipy
#Parse command line arguments, particularly an input file

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str)
parser.add_argument("output", nargs='?', default='app.log')

args = parser.parse_args()
logging.basicConfig(filename=args.output, filemode='w', format='%(message)s')
logging.getLogger().setLevel(logging.INFO)

#Parse the input file to form a molecule object
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

ops = Operator_Bank(molecule, **Get_Op_Kwargs(args.input))
logging.info(ops.cache_ops())

