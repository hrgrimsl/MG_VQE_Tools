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
#Obtain some useful global data
print('Constructing operator bank...')
print(molecule.fci_energy)
start = timer()
ops = Operator_Bank(molecule, **Get_Op_Kwargs(args.input))


try:
    v1 = v[x[0]]
except:
    pass
try:
    print('CI energy for defined space'.ljust(50)+'{}'.format(molecule.CASCI))

except:
    print('No CI energy computed.')

end = timer()
print('Operators constructed in '.ljust(50)+str(end-start)+' seconds!')
print(str(len(ops.Full_Ops))+' operations!')
#Run optimization procedure
outcome = Optimize(molecule, ops, logging, **Get_Method_Kwargs(args.input))
end = timer()

h = (ops.JW_hamiltonian)

#e,v = np.linalg.eigh(h.toarray())
#e,v = scipy.sparse.linalg.eigsh(h, k = 10)

#x = np.argsort(np.array(e))
#for i in range(0, 5):
#    casci_vec = scipy.sparse.csc_matrix(v.T[x[i]])
#    molecule.fci_energy = e[x[i]].real
#    logging.info('CASCI: '+str(molecule.fci_energy)+('S2: '+str(casci_vec.dot(ops.S2).dot(casci_vec.T.conj()).toarray()[0][0].real)))
logging.info('HF ket = '+str(ops.HF_ket))
#logging.info('HF electrons = '+str(ops.HF_ket.T.conj().dot(ops.num).dot(ops.HF_ket).toarray()[0][0].real))
#Log results
logging.info('DETCI = '+str(molecule.CASCI))
logging.info('Optimized energy = '+str(outcome.fun))
logging.info('Error = '+str(outcome.fun-molecule.CASCI))
logging.info('Iterations = '+str(outcome.nit))
logging.info('Total time (s) = '+str(end-start))
logging.info(str(len(outcome.x))+' parameters.')
print('See output in '+str(args.output)+'!')

