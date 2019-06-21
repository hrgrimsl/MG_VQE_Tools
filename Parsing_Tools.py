import inspect
import re
import os
import openfermion

import spock.core

def Get_Molecule(input_file):
    in_file = open(input_file)
    geometry = []

    loc = False
    swap = []
    skips = []
    op_kwargs = Get_Op_Kwargs(input_file)
    
    try:
        op_kwargs['active'] = [int(x) for x in op_kwargs['active'].split(',')]
    except:
        op_kwargs['active'] = None
    try:
        op_kwargs['reorder'] = [int(x) for x in op_kwargs['reorder'].split(',')]

    except:
        op_kwargs['reorder'] = None
    try:
        op_kwargs['occ'] = [int(x) for x in op_kwargs['occ'].split(',')]
    except:
        op_kwargs['occ'] = None
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
    try:
        molecule = openfermion.hamiltonians.MolecularData(geometry = geometry, basis = basis, multiplicity = multiplicity, filename = psi_file)
        molecule.load()
        molecule.active = op_kwargs['active']
        molecule.occ = op_kwargs['occ']
        molecule.n_fdoccs = n_fdoccs


        molecule.CASCI = molecule.fci_energy
        print('Loaded existing molecule.')    
        
    except:
        print('Computing new molecule.')
        molecule = spock.core.molecule(geometry = geometry, basis = basis, charge = charge, multiplicity = multiplicity, active = op_kwargs['active'], reorder = op_kwargs['reorder'], n_fdoccs = n_fdoccs, output = output, loc = loc, occ = op_kwargs['occ'])
        print(molecule.reorder)
        print(molecule.active)

        molecule = molecule.run_psi4()
        molecule.n_electrons -= 2*n_fdoccs
    molecule.filename = psi_file
    molecule.save()

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
