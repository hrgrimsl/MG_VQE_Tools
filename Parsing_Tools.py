import inspect
import re
import os
import openfermion
import openfermionpsi4
import spock
from spock import core

def Get_Molecule(input_file):
    in_file = open(input_file)
    geometry = []
    psi_file = 'temp'
    loc = False
    swap = []
    skips = []
    op_kwargs = Get_Op_Kwargs(input_file)
    op_kwargs['active'] = [int(x) for x in op_kwargs['active'].split(',')]
    op_kwargs['reorder'] = [int(x) for x in op_kwargs['reorder'].split(',')]
    for line in in_file:
        if re.search('basis', line):
            basis = line.split()[1]
        elif re.search('multiplicity', line):
            multiplicity = int(line.split()[1])
        elif re.search('charge', line):
            charge = int(line.split()[1])
        elif re.search('atom', line):
            atom = []
            full_string = line.split()
            atom.append(full_string[1])
            cart_tuple = []
            for i in range(2, 5):
                cart_tuple.append(float(full_string[i]))
            atom.append(tuple(cart_tuple))
            geometry.append(tuple(atom))
        elif re.search('psi_file', line):
            psi_file = line.split()[1]
        elif re.search('loc', line):
            loc  = line.split()[1]
        elif re.search('n_fdoccs', line):
            n_fdoccs = int(line.split()[1])
        elif re.search('swap', line):
            swap.append([int(line.split()[1]),int(line.split()[2])])
        elif re.search('output', line):
            output = line.split()[1]

    molecule = spock.core.molecule(geometry = geometry, basis = basis, charge = charge, multiplicity = multiplicity, active = op_kwargs['active'], reorder = op_kwargs['reorder'], n_fdoccs = n_fdoccs, output = output)
    molecule = molecule.run_psi4()

    
    molecule.filename = psi_file

    return molecule

def Get_Op_Kwargs(input_file):
    in_file = open(input_file)
    op_kwargs = {}
    for line in in_file:
        if re.search('op_kwarg', line):
            op_kwargs[line.split()[1]]=line.split()[2]
    return op_kwargs

def Get_Method_Kwargs(input_file):
    in_file = open(input_file)
    method_kwargs = {}
    for line in in_file:
        if re.search('method_kwarg', line):
            method_kwargs[line.split()[1]]=line.split()[2]
    return method_kwargs
