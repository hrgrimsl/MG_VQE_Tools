from evaluations import *
import copy
from Classes2 import *
import random
import numpy as np

def Optimize(molecule, ops, logging, **kwargs):
    algorithm = kwargs.get('algorithm', 'VQE')
    theta_tightness = float(kwargs.get('theta_tightness', '1e-5'))
    ADAPT_tightness = float(kwargs.get('ADAPT_tightness', '1e-5'))
    Singular_threshold = float(kwargs.get('Singular_threshold', '1e-5'))
    rfile = (kwargs.get('read', None))
    wfile = (kwargs.get('write', 'ansatz.txt'))
    seed = float(kwargs.get('RADAPT_seed', 0))
    if algorithm == 'VQE':
        parameters = []
        for i in range(0, len(ops.Full_Ops)):
            parameters.append(0)
        outcome = VQE(molecule, parameters, ops, theta_tightness, logging)
    if algorithm == 'ADAPT':
        outcome = ADAPT(molecule, ops, theta_tightness, ADAPT_tightness, logging, rfile, wfile)
    if algorithm == 'GradSort':
        parameters = []
        for i in range(0, len(ops.Full_Ops)):
            parameters.append(0)
        outcome = GradSort(molecule, parameters, ops, theta_tightness, logging)
    if algorithm == 'ADAPT_End':
        outcome = ADAPT_End(molecule, ops, theta_tightness, ADAPT_tightness, logging)
    if algorithm == 'ADAPT_All':
        outcome = ADAPT_All(molecule, ops, theta_tightness, ADAPT_tightness, logging)
    if algorithm == 'UCC':
        outcome = UCC(molecule, ops, theta_tightness, logging)
    if algorithm == 'RADAPT':
        outcome = RADAPT(molecule, ops, theta_tightness, ADAPT_tightness, logging, seed)
    if algorithm == 'LADAPT':
        outcome = LADAPT(molecule, ops, theta_tightness, ADAPT_tightness, logging)
    if algorithm == 'Force':
        outcome = Force(molecule, ops, theta_tightness, ADAPT_tightness, logging)
    if algorithm == 'ADAPT_All_Frozen_Ansatz':
        outcome = ADAPT_All_Frozen_Ansatz(molecule, ops, theta_tightness, ADAPT_tightness, logging)
    if algorithm == 'ADAPT_PT2':
        outcome = HessStep(molecule, ops, theta_tightness, ADAPT_tightness, logging, Singular_threshold)
    if algorithm == 'HOPS':
        outcome = HOPS(molecule, ops, theta_tightness, ADAPT_tightness, logging)
    if algorithm == 'VarStep':
        outcome = VarStep(molecule, ops, theta_tightness, ADAPT_tightness, logging)
    if algorithm == 'CASCI':
        wfile = open('CASCI.out', 'a')
        h = ops.JW_hamiltonian
        e,v = np.linalg.eig(h.toarray())
        wfile.write(str(e)+'\n')
        print(str(e))
        exit()
    return outcome

       
def Callback(optimization):
    print('...')

def VQE(molecule, parameters, ops, theta_tightness, logging):
    #Initialize parameters
    print('Performing optimization of parameters...')
    optimization = scipy.optimize.minimize(Trotter_SPE, parameters, jac = Trotter_Gradient, args = (ops), method = 'BFGS', options = {'gtol': float(theta_tightness), 'disp': False}, callback = Callback)
    print(str(len(parameters))+' parameters optimized in '+str(optimization.nit)+' iterations!')
    return optimization

def GradSort(molecule, parameters, ops, theta_tightness, logging):
    current_ket = ops.HF_ket
    comms = []
    hbra = ops.HF_ket.T.dot(ops.JW_hamiltonian)
    for i in range(0, len(ops.Full_JW_Ops)):
        comm = abs(2*hbra.dot(ops.Full_JW_Ops[i]).dot(current_ket).toarray()[0][0].real)
        comms.append(comm)
    idx = (np.argsort(comms))
    comms = list(np.array(comms)[idx])
    print(comms)
    ops.Full_JW_Ops = list(np.array(ops.Full_JW_Ops)[idx])
    print('Performing optimization of parameters...')
    optimization = scipy.optimize.minimize(Trotter_SPE, parameters, jac = Trotter_Gradient, args = (ops), method = 'BFGS', options = {'gtol': float(theta_tightness), 'disp': False}, callback = Callback)
    print(str(len(parameters))+' parameters optimized in '+str(optimization.nit)+' iterations!')
    return optimization

def VQE_No_H(molecule, parameters, ops, theta_tightness, logging):
    #Initialize parameters
    print('Performing optimization of parameters...')
    optimization = scipy.optimize.minimize(SPE_No_H, parameters, jac = Trotter_Gradient_No_H, args = (ops), method = 'BFGS', options = {'gtol': float(theta_tightness), 'disp': False}, callback = Callback)
    print(str(len(parameters))+' parameters optimized in '+str(optimization.nit)+' iterations!')
    print('Current energy: '+str(optimization.fun))
    return optimization

def Expand_Pool(molecule, ops, vector, ansatz):
    #Analyze gradients
    #x = number of largest gradients to pair
    x = 3
    bigops = np.array(vector).argsort()[0:x][::-1]    
    for i in list(bigops):
        for j in list(bigops):
            if j<=i:
                continue
            new_op = (ops.Full_JW_Ops[i].dot(ops.Full_JW_Ops[j])-ops.Full_JW_Ops[j].dot(ops.Full_JW_Ops[i]))
            if scipy.sparse.linalg.norm(new_op)>=1e-8:
                ops.Full_JW_Ops.append(new_op)
                ops.Full_Ops.append(ops.Full_Ops[i]+ops.Full_Ops[j])
                print(ops.Full_Ops[-1])
    return ops, vector

def ADAPT(molecule, ops, theta_tightness, ADAPT_tightness, logging, rfile, wfile):
    energy = ops.HF_ket.transpose().conj().dot(ops.JW_hamiltonian).dot(ops.HF_ket).toarray()[0][0].real
    ansatz = Ansatz_Operations(ops)
    if rfile!=None:
        ansatz.read(str(rfile), ops)
    current_ket = copy.copy(ops.HF_ket)
    parameters = ansatz.parameters
    scipy_hessians = [np.array([])]    
    gradients = [None]
    i_iter = 0
    while gradients[-1] == None or abs(gradients[-1])>ADAPT_tightness:
        grad = 0
        num = None
        hbra = current_ket.transpose().conj().dot(ops.JW_hamiltonian)
        vector = []
        for i in range(0, len(ops.Full_JW_Ops)):
            comm = 2*hbra.dot(ops.Full_JW_Ops[i]).dot(current_ket).toarray()[0][0].real
            vector.append(-abs(comm)) 
            if abs(comm)>abs(grad):
                if i in ansatz.indices and ops.repeats == 'False':
                    continue
                grad = comm
                num = i 
        if num == None:
             print('Pool exhausted.')
             break
        print('\nIteration '+str(len(parameters))+'.\n')
        '''
        print('Significant gradients:\n')
        for i in range(0, len(vector)):
            if abs(vector[i])>ADAPT_tightness:
                print('{:+10.14f}'.format(vector[i])+' {:10s}'.format(str(ops.Full_Ops[i])))
        print('\n')
        '''

        try:
            print('Next operation: {:10s}'.format(str(ops.Full_Ops[num])))
            print('Next gradient: {:+10.14f}'.format(grad))
        except:
            pass
        gradients.append(scipy.linalg.norm(vector))
        #ops, vector = Expand_Pool(molecule, ops, vector, ansatz)
        print('Norm of all gradients: {:+10.14f}'.format(gradients[-1]))

        #Comment this stuff out later
        try:
            logging.info(str(energy))
        except:
            pass
        if abs(gradients[-1])<ADAPT_tightness:
            if len(gradients) == 2:
                OptRes = scipy.optimize.OptimizeResult(x=(), fun = molecule.hf_energy, nit = 0)
            continue
        ansatz.Full_JW_Ops.insert(0, ops.Full_JW_Ops[num])
        ansatz.Full_Ops.insert(0, ops.Full_Ops[num])
        ansatz.indices.insert(0, num)

        parameters = list(parameters)
        parameters.insert(0, 0)
        ansatz.parameters = list(parameters)
        ansatz.dump(str(wfile))
        OptRes = VQE(molecule, parameters, ansatz, theta_tightness, logging)
        parameters = OptRes.x
        '''
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
        '''     
        energy = OptRes.fun
        
        scipy_hessians.append(np.linalg.pinv(OptRes.hess_inv))
        current_ket = copy.copy(ops.HF_ket)
        for i in reversed(range(0, len(parameters))):
            current_ket = scipy.sparse.linalg.expm_multiply(parameters[i]*ansatz.Full_JW_Ops[i], current_ket) 
        try:
            S2 = current_ket.transpose().conj().dot(ops.S2.dot(current_ket)).toarray()[0][0].real
            print('ADAPT S^2 = '+str(S2))
        except:
            pass
        print('Energy: {:+10.14f}'.format(energy))
        

        i_iter += 1
        

    return OptRes

def HOPS(molecule, ops, theta_tightness, ADAPT_tightness, logging):
    energy = molecule.hf_energy
    ansatz = Ansatz_Operations(ops)
    current_ket = copy.copy(ops.HF_ket)
    parameters = []
    scipy_hessians = [np.array([])]    
    gradients = [None]
    i_iter = 0
    #for op in range(0, len(ops.Full_JW_Ops)):
    #    ops.Full_JW_Ops[op] = ops.JW_hamiltonian.dot(ops.Full_JW_Ops[op])
        
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
        var = hbra.dot(ops.JW_hamiltonian).dot(current_ket).toarray()[0][0].real-(hbra.dot(current_ket).toarray()[0][0].real)**2
        print('Variance = '+str(var))
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
        try:
            logging.info(str(energy)+' '+str(gradients[-1])+' '+str(var))
        except:
            pass
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


def ADAPT_All(molecule, ops, theta_tightness, ADAPT_tightness, logging):
    Singular_threshold = 1e-10
    energy = molecule.hf_energy
    ansatz = Ansatz_Operations(ops)
    current_ket = copy.copy(ops.HF_ket)
    parameters = []
    scipy_hessians = [np.array([])]    
    gradients = [None]
    i_iter = 0
    while gradients[-1] == None or abs(gradients[-1])>ADAPT_tightness:
        #logging.info(str(energy)+' '+str(Predict_dE(molecule, ops, theta_tightness, ADAPT_tightness, logging, ansatz, parameters, energy, scipy_hessians, False, Singular_threshold)))
        grad = 0
        dE = float((Predict_dE(molecule, ops, theta_tightness, ADAPT_tightness, logging, ansatz, parameters, energy, scipy_hessians, True, Singular_threshold)))
        num = None
        hbra = current_ket.transpose().conj().dot(ops.JW_hamiltonian)
        h2ket = current_ket
        h2bra = current_ket.conj().T.dot(ops.JW_hamiltonian.dot(ops.JW_hamiltonian))
        h2 = h2bra.dot(h2ket).toarray()[0][0].real
        var = energy-abs((h2bra.dot(h2ket).toarray()[0][0].real-(current_ket.conj().T.dot(ops.JW_hamiltonian).dot(current_ket).toarray()[0][0].real)**2))
        var2 = -abs(h2/energy)
        logging.info(str(energy-molecule.fci_energy)+' '+str(energy+dE-molecule.fci_energy)+' '+str(var-molecule.fci_energy)+' '+str(var2-molecule.fci_energy))
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
        scipy_hessians.append(np.linalg.pinv(OptRes.hess_inv))
        current_ket = copy.copy(ops.HF_ket)
        for i in reversed(range(0, len(parameters))):
             current_ket = scipy.sparse.linalg.expm_multiply(parameters[i]*ansatz.Full_JW_Ops[i], current_ket) 
        i_iter += 1
    return OptRes

def ADAPT_All_Frozen_Ansatz(molecule, ops, theta_tightness, ADAPT_tightness, logging):
    energy = molecule.hf_energy
    ansatz = Ansatz_Operations(ops)
    current_ket = copy.copy(ops.HF_ket)
    parameters = []
    scipy_hessians = [np.array([])]    
    gradients = [None]
    i_iter = 0
    while gradients[-1] == None or abs(gradients[-1])>ADAPT_tightness:
        logging.info(str(energy)+' '+str(Predict_dE(molecule, ops, theta_tightness, ADAPT_tightness, logging, ansatz, parameters, energy, scipy_hessians, True, Singular_threshold)))
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
        scipy_hessians.append(np.linalg.pinv(OptRes.hess_inv))
        current_ket = copy.copy(ops.HF_ket)
        for i in reversed(range(0, len(parameters))):
             current_ket = scipy.sparse.linalg.expm_multiply(parameters[i]*ansatz.Full_JW_Ops[i], current_ket) 
        i_iter += 1
    return OptRes


def ADAPT_End(molecule, ops, theta_tightness, ADAPT_tightness, logging):
    energy = molecule.hf_energy
    ansatz = Ansatz_Operations(ops)
    current_ket = copy.copy(ops.HF_ket)
    parameters = []
    scipy_hessians = [np.array([])]    
    gradients = [None]
    i_iter = 0
    while gradients[-1] == None or abs(gradients[-1])>ADAPT_tightness:
        logging.info(Predict_dE(molecule, ops, theta_tightness, ADAPT_tightness, logging, ansatz, parameters, energy, scipy_hessians, Singular_threshold))
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
        try:
            logging.info(energy)
            logging.info(str(current_ket))
        except:
            pass
            logging.info(str(ops.HF_ket))
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
        scipy_hessians.append(np.linalg.pinv(OptRes.hess_inv))
        current_ket = copy.copy(ops.HF_ket)
        for i in reversed(range(0, len(parameters))):
             current_ket = scipy.sparse.linalg.expm_multiply(parameters[i]*ansatz.Full_JW_Ops[i], current_ket) 
        i_iter += 1
    logging.info('Final dE: '+str(Predict_dE(molecule, ops, theta_tightness, ADAPT_tightness, logging, ansatz, parameters, energy, scipy_hessians, Singular_threshold)))
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
        try:
            logging.info(str(energy))
        except:
            pass
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
        logging.info(str(energy))
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

def Force(molecule, ops, theta_tightness, ADAPT_tightness, logging):
    energy = molecule.hf_energy 
    OptRes = scipy.optimize.OptimizeResult(x=(), fun = molecule.hf_energy, nit = 0)
    scipy_hessians = [np.array([])]    
    ansatz = Ansatz_Operations(ops)
    parameters = []
    hessian_norm = None
    max = None
    while max == None or max>ADAPT_tightness:
        hessian, gradient = Build_Hessian(ansatz, ops, molecule, parameters, scipy_hessians)
        hessian_norm = np.linalg.norm(hessian)
        if len(parameters)>0:            
            gradient = np.hstack((gradient, np.zeros((len(parameters)))))
        hinv = -np.linalg.pinv(hessian)
        grad = (gradient[:len(ops.Full_JW_Ops)])
        dx = -hinv.dot(gradient.transpose())
        dE = .5*gradient.dot(dx)
        #grad = gradient
        max = 0
        num = None
        for i in range(0, len(grad)):
            if abs(dx[i])>abs(max):
                num = i
                max = dx[i]
        ansatz.Full_JW_Ops.insert(0, ops.Full_JW_Ops[num])
        ansatz.Full_Ops.insert(0, ops.Full_Ops[num])
        parameters = list(parameters)
        parameters.insert(0, 0)
        dx2 = [dx[i]]+dx[len(ops.Full_JW_Ops):]
        if dE>0:
            dE = -dE
            dx2 = -np.array(dx2)
        parameters = np.array(parameters)+np.array(dx2)
        print('\nIteration '+str(len(parameters))+'.\n')
        print('Next operation: {:10s}'.format(str(ops.Full_Ops[num])))
        logging.info('Predicted energy: {:10.14f}'.format(energy+dE))
        OptRes = VQE(molecule, parameters, ansatz, theta_tightness, logging)
        parameters = OptRes.x
        energy = OptRes.fun
        scipy_hessians.append(np.linalg.pinv(OptRes.hess_inv))
    return OptRes
        

def VarStep(molecule, ops, theta_tightness, ADAPT_tightness, logging):
    energy = molecule.hf_energy
    current_ket = copy.copy(ops.HF_ket)
    ansatz = Ansatz_Operations(ops)
    tot_var = 0
    H = ops.JW_hamiltonian
    H2 = H.dot(H)
    #var = current_ket.T.conj().dot(H2).dot(current_ket).toarray()[0][0].real-(current_ket.T.conj().dot(H).dot(current_ket).toarray()[0][0].real)**2    
    var = np.sqrt(abs(current_ket.T.conj().dot(H2).dot(current_ket).toarray()[0][0].real))
    print('Current energy: '+str(molecule.hf_energy))
    print('Energy prediction: '+str(-abs(var)))   
    print('CI energy: '+str(molecule.fci_energy))
    ofile = open('varstep_pes', 'a')
    ofile.write(str(molecule.hf_energy-molecule.fci_energy)+' '+str(molecule.mp2_energy-molecule.fci_energy)+' '+str(-abs(var)-molecule.fci_energy)+'\n')
    exit()

def HessStep(molecule, ops, theta_tightness, ADAPT_tightness, logging, Singular_threshold):
    energy = molecule.hf_energy
    current_ket = copy.copy(ops.HF_ket)
    ansatz = Ansatz_Operations(ops)
    parameters = []
    scipy_hessians = []
    dE = float((Predict_dE(molecule, ops, theta_tightness, ADAPT_tightness, logging, ansatz, parameters, energy, scipy_hessians, True, Singular_threshold)))
    print('Current energy: '+str(molecule.hf_energy))
    print('Energy prediction: '+str(energy-abs(dE)))   
    print('CI energy: '+str(molecule.fci_energy))
    ofile = open('varstep_pes', 'a')
    ofile.write(str(molecule.hf_energy-molecule.fci_energy)+' '+str(molecule.mp2_energy-molecule.fci_energy)+' '+str(energy-abs(dE)-molecule.fci_energy)+'\n')
    exit()
