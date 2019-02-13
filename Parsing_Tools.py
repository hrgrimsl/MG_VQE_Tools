import re
import openfermion
import openfermionpsi4

def Get_Molecule(input_file):
    in_file = open(input_file)
    geometry = []
    psi_file = 'temp'
    loc = False
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
        elif re.search('nfd', line):
            nfd = line.split()[1]
    molecule = openfermion.hamiltonians.MolecularData(geometry, basis, multiplicity, charge)
    molecule.filename = psi_file
    molecule = openfermionpsi4.run_psi4(molecule, run_scf = 1, run_mp2 = 1, run_ccsd = 1, run_cisd = 1, run_fci = 1, localize = loc)
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
