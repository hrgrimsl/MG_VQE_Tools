from evaluations import *
import copy
from Classes import *


def Optimize(molecule, ops, logging, **kwargs):
    algorithm = kwargs.get('algorithm', 'VQE')
    theta_tightness = float(kwargs.get('theta_tightness', '1e-5'))
    ADAPT_tightness = float(kwargs.get('ADAPT_tightness', '1e-5'))
    if algorithm == 'VQE':
        outcome = VQE(molecule, ops, theta_tightness)
    if algorithm == 'ADAPT':
        outcome = ADAPT(molecule, ops, theta_tightness, ADAPT_tightness, logging)
    return outcome
def VQE(molecule, ops, theta_tightness):
    #Initialize parameters
    parameters = []
    for i in range(0, len(ops.Full_Ops)):
        parameters.append(0)
    print('Performing optimization of parameters...')
    optimization = scipy.optimize.minimize(Trotter_SPE, parameters, jac = Trotter_Gradient, args = (ops), method = 'BFGS', options = {'gtol': float(theta_tightness), 'disp': False})
    print(str(len(parameters))+' parameters optimized in '+str(optimization.nit)+' iterations!')
    return optimization

def ADAPT(molecule, ops, theta_tightness, ADAPT_tightness, logging):
    ansatz = Ansatz_Operations(ops)
    current_ket = copy.copy(ops.HF_ket)
    parameters = []
    gradients = [None]
    while gradients[-1] == None or abs(gradients[-1])>ADAPT_tightness:
        grad = 0
        num = None
        hbra = current_ket.transpose().dot(ops.JW_hamiltonian)
        for i in range(0, len(ops.Full_JW_Ops)):
            comm = 2*hbra.dot(ops.Full_JW_Ops[i]).dot(current_ket).toarray()[0][0].real
            if abs(comm)>abs(grad):
                grad = comm
                num = i
        print('Next gradient: '+str(grad))
        print('Next operation: '+str(ops.Full_Ops[num]))
        gradients.append(grad)
        ansatz.Full_JW_Ops.insert(0, ops.Full_JW_Ops[num])
        ansatz.Full_Ops.insert(0, ops.Full_Ops[num])
        OptRes = VQE(molecule, ansatz, theta_tightness)
        parameters = OptRes.x
        energy = OptRes.fun
        current_ket = copy.copy(ops.HF_ket)
        for i in reversed(range(0, len(parameters))):
             current_ket = scipy.sparse.linalg.expm_multiply(parameters[i]*ansatz.Full_JW_Ops[i], current_ket) 
    return OptRes    
