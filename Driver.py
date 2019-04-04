#!/usr/bin/env python3
from Parsing_Tools import *
from Classes2 import *
import os
import argparse
import logging
import openfermion
import scipy
from methods import *
from timeit import default_timer as timer
import hat

#Parse command line arguments, particularly an input file

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

logging.info('HF = '+str(molecule.hf_energy))




#Obtain some useful global data

print('Constructing operator bank...')
start = timer()
ops = Operator_Bank(molecule, **Get_Op_Kwargs(args.input))
h = (ops.JW_hamiltonian)
if h.shape[0]<128:
    e = np.linalg.eigh(h.toarray())[0]
else:
    e,v = (scipy.sparse.linalg.eigsh(h, k = 1, v0 = ops.HF_ket.toarray()))
x = np.argsort(np.array(e))
molecule.fci_energy = e[x[0]].real
try:
    v1 = v[x[0]]
    print(scipy.sparse.csc_matrix(v))
except:
    pass
print('CI energy for defined space'.ljust(50)+'{}'.format(molecule.fci_energy))

logging.info('CASCI = '+str(molecule.fci_energy))
end = timer()
print('Operators constructed in '.ljust(50)+str(end-start)+' seconds!')
print(str(len(ops.Full_Ops))+' operations!')
#Run optimization procedure
outcome = Optimize(molecule, ops, logging, **Get_Method_Kwargs(args.input))
end = timer()

#Log results
logging.info('Optimized energy = '+str(outcome.fun))
logging.info('Error = '+str(outcome.fun-molecule.fci_energy))
logging.info('Iterations = '+str(outcome.nit))
logging.info('Total time (s) = '+str(end-start))
logging.info(str(len(outcome.x))+' parameters.')
print('See output in '+str(args.output)+'!')

