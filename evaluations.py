import scipy
import copy
import numpy as np

def Trotter_SPE(parameters, ops):
    ket = copy.copy(ops.HF_ket)
    for i in reversed(range(0, len(parameters))):
        ket = scipy.sparse.linalg.expm_multiply(ops.Full_JW_Ops[i]*parameters[i], ket)
    return ket.transpose().conj().dot(ops.JW_hamiltonian).dot(ket).toarray()[0][0].real

def SPE(parameters, ops):
    generator = ops.Full_JW_Ops[-1]
    for i in reversed(range(0, len(parameters))):
        generator+=ops.Full_JW_Ops[i]*parameters[i]
    ket = scipy.sparse.linalg.expm_multiply(generator, ops.HF_ket)
    return ket.transpose().dot(ops.JW_hamiltonian).dot(ket).toarray()[0][0].real

def Trotter_Gradient(parameters, ops):
    grad = []
    ket = copy.copy(ops.HF_ket)
    for i in reversed(range(0, len(parameters))):
        ket = scipy.sparse.linalg.expm_multiply(ops.Full_JW_Ops[i]*parameters[i], ket)
    hbra = ket.transpose().conj().dot(ops.JW_hamiltonian)
    term = 0 
    grad = Recurse(parameters, grad, hbra, ket, term, ops)
    return np.array(grad)

def Recurse(parameters, grad, hbra, ket, term, ops):
    if term != 0:
        hbraket = scipy.sparse.hstack([hbra.transpose().conj(), ket])
        hbraket = scipy.sparse.csc_matrix(scipy.sparse.linalg.expm_multiply(-ops.Full_JW_Ops[term-1]*parameters[term-1], hbraket))
        hbra = hbraket.transpose()[0,:].conj()
        ket = hbraket.transpose()[1,:].transpose()
    grad.append(2*hbra.dot(ops.Full_JW_Ops[term]).dot(ket).toarray()[0][0].real)
    if term<len(parameters)-1:
        term+=1
        Recurse(parameters, grad, hbra, ket, term, ops)
    return np.asarray(grad)

    
