from evaluations import *
import copy
from Classes import *
import random


def Optimize(molecule, ops, logging, **kwargs):
    algorithm = kwargs.get('algorithm', 'VQE')
    theta_tightness = float(kwargs.get('theta_tightness', '1e-5'))
    ADAPT_tightness = float(kwargs.get('ADAPT_tightness', '1e-5'))
    seed = float(kwargs.get('RADAPT_seed', 0))
    if algorithm == 'VQE':
        outcome = VQE(molecule, ops, theta_tightness)
    if algorithm == 'ADAPT':
        outcome = ADAPT(molecule, ops, theta_tightness, ADAPT_tightness, logging)
    if algorithm == 'UCC':
        outcome = UCC(molecule, ops, theta_tightness)
    if algorithm == 'RADAPT':
        outcome = RADAPT(molecule, ops, theta_tightness, ADAPT_tightness, logging, seed)
    if algorithm == 'LADAPT':
        outcome = LADAPT(molecule, ops, theta_tightness, ADAPT_tightness, logging)

    return outcome

def VQE(molecule, ops, theta_tightness):
    #Initialize parameters
    parameters = []
    for i in range(0, len(ops.Full_Ops)):
        parameters.append(0)
    print('Performing optimization of parameters...')
    optimization = scipy.optimize.minimize(Trotter_SPE, parameters, jac = Trotter_Gradient, args = (ops), method = 'BFGS', options = {'gtol': float(theta_tightness), 'disp': False})
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
<<<<<<< HEAD
<<<<<<< HEAD
                num = i 
        print('\nIteration '+str(len(parameters))+'.\n')
        print('Significant gradients:\n')
        for i in range(0, len(vector)):
            if abs(vector[i])>ADAPT_tightness:
                print(str(vector[i])+' '+str(ops.Full_Ops[i]))
        print('\n')
=======
=======
>>>>>>> 95c740359aedaad677411e8004e40767affa7777
                num = i
    
        print()
        print(" -----------------------------------------------------------------")
        print('         ADAPT-VQE Iteration %i' %i_iter )
        print(" -----------------------------------------------------------------")
<<<<<<< HEAD
>>>>>>> 95c740359aedaad677411e8004e40767affa7777
=======
>>>>>>> 95c740359aedaad677411e8004e40767affa7777
        print('Next operation: '+str(ops.Full_Ops[num]))
        print('Next gradient: '+str(grad))
        gradients.append(scipy.linalg.norm(vector))
        print('Norm of all gradients: '+str(gradients[-1]))
        if abs(gradients[-1])<ADAPT_tightness:
            continue
        ansatz.Full_JW_Ops.insert(0, ops.Full_JW_Ops[num])
        print('Newest full ansatz:\n')
        ansatz.Full_Ops.insert(0, ops.Full_Ops[num])
        for term in ansatz.Full_Ops:
            string = ''
<<<<<<< HEAD
            for subterm in range(0, len(term)):
                if subterm%2 == 0:
                    string+=str(int(term[subterm]))+'^t '
                else:
                    string+=str(int(term[subterm]))+' '
=======
            for subterm in term:
                if term.index(subterm)%2 == 0:
                    string+="%3i'"%(int(subterm))
                else:
                    string+="%3i"%(int(subterm))
<<<<<<< HEAD
>>>>>>> 95c740359aedaad677411e8004e40767affa7777
=======
>>>>>>> 95c740359aedaad677411e8004e40767affa7777
            print(string)
        print('\n')        
        OptRes = VQE(molecule, ansatz, theta_tightness)
        parameters = OptRes.x
        energy = OptRes.fun
        current_ket = copy.copy(ops.HF_ket)
        for i in reversed(range(0, len(parameters))):
             current_ket = scipy.sparse.linalg.expm_multiply(parameters[i]*ansatz.Full_JW_Ops[i], current_ket) 

        i_iter += 1
    return OptRes

def RADAPT(molecule, ops, theta_tightness, ADAPT_tightness, logging, seed):
    random.seed()
    ansatz = Ansatz_Operations(ops)
    current_ket = copy.copy(ops.HF_ket)
    parameters = []
    gradients = [None]
    while gradients[-1] == None or abs(gradients[-1])>ADAPT_tightness:
        grad = 0
        num = None
        hbra = current_ket.transpose().conj().dot(ops.JW_hamiltonian)
        vector = []
        num = random.randint(0, len(ops.Full_JW_Ops))
        for i in range(0, len(ops.Full_JW_Ops)):
            comm = 2*hbra.dot(ops.Full_JW_Ops[i]).dot(current_ket).toarray()[0][0].real
            vector.append(comm) 
            if i == num:
                grad = comm
        print('\nIteration '+str(len(parameters))+'.\n')
        print('Next operation: '+str(ops.Full_Ops[num]))
        print('Next gradient: '+str(grad))
        gradients.append(scipy.linalg.norm(vector))
        print('Norm of all gradients: '+str(gradients[-1]))
        if abs(gradients[-1])<ADAPT_tightness:
            continue
        print('Newest full ansatz:\n')
        for term in ansatz.Full_Ops:
            string = ''
            for subterm in range(0, len(term)):
                if subterm%2 == 0:
                    string+=str(int(term[subterm]))+'^t '
                else:
                    string+=str(int(term[subterm]))+' '
            print(string)
        print('\n')        
        ansatz.Full_JW_Ops.insert(0, ops.Full_JW_Ops[num])
        ansatz.Full_Ops.insert(0, ops.Full_Ops[num])
        OptRes = VQE(molecule, ansatz, theta_tightness)
        parameters = OptRes.x
        energy = OptRes.fun
        current_ket = copy.copy(ops.HF_ket)
        for i in reversed(range(0, len(parameters))):
             current_ket = scipy.sparse.linalg.expm_multiply(parameters[i]*ansatz.Full_JW_Ops[i], current_ket) 
    return OptRes

def LADAPT(molecule, ops, theta_tightness, ADAPT_tightness, logging):
    random.seed()
    ansatz = Ansatz_Operations(ops)
    current_ket = copy.copy(ops.HF_ket)
    parameters = []
    gradients = [None]
    num = 0
    while (gradients[-1] == None or abs(gradients[-1])>ADAPT_tightness) and num<len(ops.Full_JW_Ops):
        grad = 0
        hbra = current_ket.transpose().conj().dot(ops.JW_hamiltonian)
        vector = []
        for i in range(0, len(ops.Full_JW_Ops)):
            comm = 2*hbra.dot(ops.Full_JW_Ops[i]).dot(current_ket).toarray()[0][0].real
            vector.append(comm) 
            if i == num:
                grad = comm
        print('\nIteration :'+str(len(parameters))+'\n')
        print('Next operation: '+str(ops.Full_Ops[num]))
        print('Next gradient: '+str(grad))
        gradients.append(scipy.linalg.norm(vector))
        print('Norm of all gradients: '+str(gradients[-1]))
        if abs(gradients[-1])<ADAPT_tightness or num==len(ops.Full_JW_Ops):
            continue
        ansatz.Full_JW_Ops.insert(0, ops.Full_JW_Ops[num])
        ansatz.Full_Ops.insert(0, ops.Full_Ops[num])
        print('Full ansatz: \n')
        for term in ansatz.Full_Ops:
            string = ''
            for subterm in range(0, len(term)):
                if subterm%2 == 0:
                    string+=str(int(term[subterm]))+'^t '
                else:
                    string+=str(int(term[subterm]))+' '
            print(string)
        print('\n')        
        OptRes = VQE(molecule, ansatz, theta_tightness)
        parameters = OptRes.x
        energy = OptRes.fun
        current_ket = copy.copy(ops.HF_ket)
        for i in reversed(range(0, len(parameters))):
             current_ket = scipy.sparse.linalg.expm_multiply(parameters[i]*ansatz.Full_JW_Ops[i], current_ket) 
        num+=1
    return OptRes

def UCC(molecule, ops, theta_tightness):
    #Initialize parameters
    global last
    last = 0
    parameters = [0 for op in ops.Full_Ops]
    print('Performing optimization of parameters...')
    optimization = scipy.optimize.minimize(UCC_SPE, parameters, args = (ops), method = 'BFGS', options = {'gtol': float(theta_tightness), 'disp': False}) 
    print(str(len(parameters))+' parameters optimized in '+str(optimization.nit)+' iterations!')
    return optimization

