from evaluations import *
import copy
from Classes2 import *
import random


def Optimize(molecule, ops, logging, **kwargs):
    algorithm = kwargs.get('algorithm', 'VQE')
    theta_tightness = float(kwargs.get('theta_tightness', '1e-5'))
    ADAPT_tightness = float(kwargs.get('ADAPT_tightness', '1e-5'))
    seed = float(kwargs.get('RADAPT_seed', 0))
    if algorithm == 'VQE':
        parameters = []
        for i in range(0, len(ops.Full_Ops)):
            parameters.append(0)
        outcome = VQE(molecule, parameters, ops, theta_tightness, logging)
    if algorithm == 'ADAPT':
        outcome = ADAPT(molecule, ops, theta_tightness, ADAPT_tightness, logging)
    if algorithm == 'UCC':
        outcome = UCC(molecule, ops, theta_tightness, logging)
    if algorithm == 'RADAPT':
        outcome = RADAPT(molecule, ops, theta_tightness, ADAPT_tightness, logging, seed)
    if algorithm == 'LADAPT':
        outcome = LADAPT(molecule, ops, theta_tightness, ADAPT_tightness, logging)

    return outcome
def Callback(optimization):
    print(optimization)
def VQE(molecule, parameters, ops, theta_tightness, logging):
    #Initialize parameters
    print('Performing optimization of parameters...')
    optimization = scipy.optimize.minimize(Trotter_SPE, parameters, jac = Trotter_Gradient, args = (ops), method = 'BFGS', options = {'gtol': float(theta_tightness), 'disp': False}, callback = Callback)
    print(str(len(parameters))+' parameters optimized in '+str(optimization.nit)+' iterations!')
    print('Current energy: '+str(optimization.fun))
    return optimization

def ADAPT(molecule, ops, theta_tightness, ADAPT_tightness, logging):
    ansatz = Ansatz_Operations(ops)
    current_ket = copy.copy(ops.HF_ket)
    parameters = []
    gradients = [None]
    i_iter = 0
    while gradients[-1] == None or abs(gradients[-1])>ADAPT_tightness:
        grad = 0
        num = None
        hbra = current_ket.transpose().conj().dot(ops.JW_hamiltonian)
        vector = []
        for i in range(0, len(ops.Full_JW_Ops)):
            comm = 2*hbra.dot(ops.Full_JW_Ops[i]).dot(current_ket).toarray()[0][0].real
            vector.append(comm) 
            if abs(comm)>abs(grad):
                grad = comm
                num = i 
        print('\nIteration '+str(len(parameters))+'.\n')
        print('Significant gradients:\n')
        for i in range(0, len(vector)):
            if abs(vector[i])>ADAPT_tightness:
                print('{:+10.14f}'.format(vector[i])+' {:10s}'.format(str(ops.Full_Ops[i])))
        print('\n')
        print('Next operation: {:10s}'.format(str(ops.Full_Ops[num])))
        print('Next gradient: {:+10.14f}'.format(grad))
        gradients.append(scipy.linalg.norm(vector))
        print('Norm of all gradients: {:+10.14f}'.format(gradients[-1]))


        #Comment this stuff out later


        if abs(gradients[-1])<ADAPT_tightness:
            if len(gradients) == 2:
                OptRes = scipy.optimize.OptimizeResult(x=(), fun = molecule.hf_energy, nit = 0)
            continue
        ansatz.Full_JW_Ops.insert(0, ops.Full_JW_Ops[num])
        ansatz.Full_Ops.insert(0, ops.Full_Ops[num])
        parameters = list(parameters)
        parameters.insert(0, 0)
        OptRes = VQE(molecule, parameters, ansatz, theta_tightness, logging)
        parameters = OptRes.x
        print('Newest full ansatz:\n')
        for term in range(0, len(ansatz.Full_Ops)):
            string = '%+14.10f'%float(parameters[term])+' '
            for subterm in range(0, len(ansatz.Full_Ops[term])):
                if subterm%2 == 0:
                    string+="%3i'" %int(ansatz.Full_Ops[term][subterm])
                else:
                    string+="%3i" %int(ansatz.Full_Ops[term][subterm])
            print(string)
        print('\n')        
        energy = OptRes.fun
        current_ket = copy.copy(ops.HF_ket)
        for i in reversed(range(0, len(parameters))):
             current_ket = scipy.sparse.linalg.expm_multiply(parameters[i]*ansatz.Full_JW_Ops[i], current_ket) 

        i_iter += 1
    return OptRes

def RADAPT(molecule, ops, theta_tightness, ADAPT_tightness, logging, seed):
    assert(len(ops.Full_Ops)==len(ops.Full_JW_Ops))
    ansatz = Ansatz_Operations(ops)
    current_ket = copy.copy(ops.HF_ket)
    parameters = []
    gradients = [None]

    i_iter = 0
    while gradients[-1] == None or abs(gradients[-1])>ADAPT_tightness:
        grad = 0
        num = None
        hbra = current_ket.transpose().conj().dot(ops.JW_hamiltonian)
        vector = []
        for i in range(0, len(ops.Full_JW_Ops)):
            comm = 2*hbra.dot(ops.Full_JW_Ops[i]).dot(current_ket).toarray()[0][0].real
            vector.append(comm) 
            if num == i:
                grad = comm
        num = random.randint(0, len(ops.Full_Ops)-1)
        print('\nIteration '+str(len(parameters))+'.\n')
        print('Significant gradients:\n')
        for i in range(0, len(vector)):
            if abs(vector[i])>ADAPT_tightness:
                print('{:+10.14f}'.format(vector[i])+' {:10s}'.format(str(ops.Full_Ops[i])))
        print('\n')
        print('Next operation: {:10s}'.format(str(ops.Full_Ops[num])))
        print('Next gradient: {:+10.14f}'.format(grad))
        gradients.append(scipy.linalg.norm(vector))
        print('Norm of all gradients: {:+10.14f}'.format(gradients[-1]))
        if abs(gradients[-1])<ADAPT_tightness:
            if len(gradients) == 2:
                OptRes = scipy.optimize.OptimizeResult(x=(), fun = molecule.hf_energy, nit = 0)
            continue
        ansatz.Full_JW_Ops.insert(0, ops.Full_JW_Ops[num])
        ansatz.Full_Ops.insert(0, ops.Full_Ops[num])
        parameters = list(parameters)
        parameters.insert(0, 0)

        OptRes = VQE(molecule, parameters, ansatz, theta_tightness, logging)
        parameters = OptRes.x
        print('Newest full ansatz:\n')
        for term in range(0, len(ansatz.Full_Ops)):
            string = '%+14.10f'%float(parameters[term])+' '
            for subterm in range(0, len(ansatz.Full_Ops[term])):
                if subterm%2 == 0:
                    string+="%3i'" %int(ansatz.Full_Ops[term][subterm])
                else:
                    string+="%3i" %int(ansatz.Full_Ops[term][subterm])
            print(string)
        print('\n')        
        energy = OptRes.fun
        current_ket = copy.copy(ops.HF_ket)
        for i in reversed(range(0, len(parameters))):
             current_ket = scipy.sparse.linalg.expm_multiply(parameters[i]*ansatz.Full_JW_Ops[i], current_ket) 

        i_iter += 1
    return OptRes

def LADAPT(molecule, ops, theta_tightness, ADAPT_tightness, logging):
    ansatz = Ansatz_Operations(ops)
    current_ket = copy.copy(ops.HF_ket)
    parameters = []
    gradients = [None]

    i_iter = 0
    num = 0
    while gradients[-1] == None or abs(gradients[-1])>ADAPT_tightness:
        grad = 0
        hbra = current_ket.transpose().conj().dot(ops.JW_hamiltonian)
        vector = []
        for i in range(0, len(ops.Full_JW_Ops)):
            comm = 2*hbra.dot(ops.Full_JW_Ops[i]).dot(current_ket).toarray()[0][0].real
            vector.append(comm) 
            if num == i:
                grad = comm

        print('\nIteration '+str(len(parameters))+'.\n')
        print('Significant gradients:\n')
        for i in range(0, len(vector)):
            if abs(vector[i])>ADAPT_tightness:
                print('{:+10.14f}'.format(vector[i])+' {:10s}'.format(str(ops.Full_Ops[i])))
        print('\n')
        print('Next operation: {:10s}'.format(str(ops.Full_Ops[num])))
        print('Next gradient: {:+10.14f}'.format(grad))
        gradients.append(scipy.linalg.norm(vector))
        print('Norm of all gradients: {:+10.14f}'.format(gradients[-1]))
        if abs(gradients[-1])<ADAPT_tightness:
            if len(gradients) == 2:
                OptRes = scipy.optimize.OptimizeResult(x=(), fun = molecule.hf_energy, nit = 0)
            continue
        ansatz.Full_JW_Ops.insert(0, ops.Full_JW_Ops[num])
        ansatz.Full_Ops.insert(0, ops.Full_Ops[num])
        parameters = list(parameters)

        parameters.insert(0, 0)
       

        OptRes = VQE(molecule, parameters, ansatz, theta_tightness, logging)
        parameters = OptRes.x
        print('Newest full ansatz:\n')
        for term in range(0, len(ansatz.Full_Ops)):
            string = '%+14.10f'%float(parameters[term])+' '
            for subterm in range(0, len(ansatz.Full_Ops[term])):
                if subterm%2 == 0:
                    string+="%3i'" %int(ansatz.Full_Ops[term][subterm])
                else:
                    string+="%3i" %int(ansatz.Full_Ops[term][subterm])
            print(string)
        print('\n')        
        energy = OptRes.fun
        print(energy)
        current_ket = copy.copy(ops.HF_ket)
        for i in reversed(range(0, len(parameters))):
             current_ket = scipy.sparse.linalg.expm_multiply(parameters[i]*ansatz.Full_JW_Ops[i], current_ket) 

        i_iter += 1
        num += 1
    return OptRes

def UCC(molecule, ops, theta_tightness, logging):
    parameters = []
    for i in range(0, len(ops.Full_Ops)):
        parameters.append(0)
    optimization = scipy.optimize.minimize(UCC_SPE, parameters, args = (ops), method = 'BFGS', options = {'gtol': float(theta_tightness), 'disp': False})
    return optimization
    
