import scipy
import numpy as np
import openfermioncirq
import cirq
import copy

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

def SPE_No_H(parameters, ops):
    ket = copy.copy(ops.HF_ket)
    for i in reversed(range(0, len(parameters))):
        ket = scipy.sparse.linalg.expm_multiply(ops.Full_JW_Ops[i]*parameters[i], ket)
    return ket.transpose().conj().dot(ket).toarray()[0][0].real

def Numerical_Hessian(ansatz, ops, molecule, parameters, scipy_hessians):
    Hessian = []
    operations = ops.Full_JW_Ops+ansatz.Full_JW_Ops
    params = []
    for i in range(0, len(ops.Full_JW_Ops)):
        params.append(0)
    params += list(parameters)
    for parameter1 in range(0, len(params)):
        grad = (scipy.optimize.approx_fprime(params, Numerical_Gradient, 1e-6, operations, parameter1, ops, ansatz))
        Hessian.append(grad)
    return (np.array(Hessian))

def Numerical_Gradient(params, operations, parameter1, ops, ansatz):
    grad = scipy.optimize.approx_fprime(params, Numerical_Energy, 1e-6, operations, ops)
    return grad[parameter1]

def Numerical_Energy(params, operations, ops):
    ket = copy.copy(ops.HF_ket)
    for i in reversed(range(0, len(params))):
        ket = scipy.sparse.linalg.expm_multiply(params[i]*operations[i], ket)
    energy = ket.transpose().conj().dot(ops.JW_hamiltonian).dot(ket)
    return energy.toarray()[0][0].real           

def Trotter_Gradient(parameters, ops):
    grad = []
    ket = copy.copy(ops.HF_ket)
    for i in reversed(range(0, len(parameters))):
        ket = scipy.sparse.linalg.expm_multiply(ops.Full_JW_Ops[i]*parameters[i], ket)
    hbra = ket.transpose().conj().dot(ops.JW_hamiltonian)
    term = 0 
    grad = Recurse(parameters, grad, hbra, ket, term, ops)
    return np.array(grad)

def Trotter_Gradient_No_H(parameters, ops):
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

def UCC_SPE(parameters, ops):
    ket = ops.HF_ket
    gen = ops.Full_JW_Ops[0]*parameters[0]
    for i in range(1, len(parameters)):
        gen+=ops.Full_JW_Ops[i]*parameters[i]
    ket = scipy.sparse.linalg.expm_multiply(gen, ket)
    energy = ket.transpose().dot(ops.JW_hamiltonian).dot(ket).toarray()[0][0].real
    return energy

def Predict_dE(molecule, ops, theta_tightness, ADAPT_tightness, logging, ansatz, parameters, energy, scipy_hessians, frozen_ansatz, Singular_threshold):
    if frozen_ansatz == False:
        hessian, gradient = Build_Hessian(ansatz, ops, molecule, parameters, scipy_hessians)
    else:
        hessian, gradient = Build_NW_Hessian_Only(ansatz, ops, molecule, parameters, scipy_hessians)
    hessian_norm = np.linalg.norm(hessian)
    if len(parameters)>0 and frozen_ansatz == False:    
        gradient = np.hstack((gradient, np.zeros((len(parameters)))))
    print(frozen_ansatz)
    grad = (gradient[:len(ops.Full_JW_Ops)])
    gradient = list(gradient)
    grad2 = []
    for i in gradient:
        grad2.append([i])
    gradient = np.array(grad2)
    u, s, v = np.linalg.svd(hessian, full_matrices=True, compute_uv=True) 
    for i in list(s):
        S = []
        if i>=Singular_threshold:
            S.append(i)
    s2 = np.diag(np.array(S))
    u2 = u[:,:len(s2)]
    v2 = v[:len(s2),:]
    hessian = u2.dot(s2).dot(v2)
    sinv = []
    for j in range(0, len(s2[-1])):
        sinv.append(1/S[j])
    sinv = np.diag(np.array(sinv))
    hinv = v2.transpose().conj().dot(sinv).dot(u2.transpose().conj())
    dx = -hinv.dot(gradient)
    dE = gradient.conj().transpose().dot(dx)+.5*dx.conj().transpose().dot(hessian).dot(dx)
    dE = float(dE)
    print('dE = '+str(dE))
    logging.info(str(energy+dE))
    #grad = gradient
    if dE>0:
        dE = -dE 
    return dE

def Build_NW_Hessian_Only(ansatz, ops, molecule, parameters, scipy_hessians):
    cur_ket = ops.HF_ket
    for i in reversed(range(0, len(parameters))):
        cur_ket = scipy.sparse.linalg.expm_multiply(parameters[i]*ansatz.Full_JW_Ops[i], cur_ket)
    hessian = {}
    gradient = []
    lg = np.zeros((len(ops.Full_JW_Ops),len(ops.Full_JW_Ops)))
    for a in range(0, len(ops.Full_JW_Ops)):
        comm = ops.JW_hamiltonian.dot(ops.Full_JW_Ops[a])
        comm-=ops.Full_JW_Ops[a].dot(ops.JW_hamiltonian)
        gradient.append(cur_ket.transpose().conj().dot(comm).dot(cur_ket).toarray()[0][0].real)
        for b in range(0, len(ops.Full_JW_Ops)):
            lg[a][b] = (2*cur_ket.transpose().conj().dot(comm).dot(ops.Full_JW_Ops[b]).dot(cur_ket).toarray()[0][0].real)

    hessian = np.array(lg)
    return hessian, np.array(gradient)

def Build_Hessian(ansatz, ops, molecule, parameters, scipy_hessians):
    cur_ket = ops.HF_ket
    for i in reversed(range(0, len(parameters))):
        cur_ket = scipy.sparse.linalg.expm_multiply(parameters[i]*ansatz.Full_JW_Ops[i], cur_ket)
    hessian = {}
    #A, B in pool
    lg = np.zeros((len(ops.Full_JW_Ops),len(ops.Full_JW_Ops)))
    for a in range(0, len(ops.Full_JW_Ops)):
        for b in range(a, len(ops.Full_JW_Ops)):
            comm = ops.JW_hamiltonian.dot(ops.Full_JW_Ops[a])
            comm-=ops.Full_JW_Ops[a].dot(ops.JW_hamiltonian)
            lg[a][b] = (2*cur_ket.transpose().conj().dot(comm).dot(ops.Full_JW_Ops[b]).dot(cur_ket).toarray()[0][0].real)
            lg[b][a] = lg[b][a]
    hessian['lg'] = np.array(lg)
    #A in pool, B not
    gradient = []
    for a in range(0, len(ops.Full_JW_Ops)):
        HAbra = ops.JW_hamiltonian.dot(ops.Full_JW_Ops[a])
        HAbra-= ops.Full_JW_Ops[a].dot(ops.JW_hamiltonian)
        HAbra = (cur_ket).transpose().conj().dot(HAbra)
        N = 0
        cg = np.zeros((len(ops.Full_JW_Ops),len(ansatz.Full_JW_Ops)))
        hbra = ops.JW_hamiltonian.dot(ops.Full_JW_Ops[a]).dot(cur_ket).transpose().conj()
        cg, gradient = CG_Hessian(gradient, ansatz, ops, molecule, parameters, N, a, HAbra, hbra, cur_ket, copy.copy(cur_ket), cg)
    hessian['cg'] = cg
    
    hessian['ce'] = scipy_hessians[-1]
    #A, B, not in pool
    hessian['le'] = np.zeros((cg.shape[-1],cg.shape[-2]))
    for i in range(0, cg.shape[0]):
        for j in range(0, cg.shape[1]):
            hessian['le'][j][i]=cg[i][j]
    if len(parameters)>0:
        law = np.vstack((hessian['lg'], hessian['le']))
        chaos = np.vstack((hessian['cg'],hessian['ce']))
        hessian = np.hstack((law,chaos))
    else:
        hessian = hessian['lg']
    return hessian, np.array(gradient)

def CG_Hessian(gradient, ansatz, ops, molecule, parameters, N, a, HAbra, hbra, ket, cur_ket, cg):
    if N!=0:
        hbraket = scipy.sparse.hstack((HAbra.transpose().conj(), ket))
        hbraket = scipy.sparse.hstack((hbraket, hbra.transpose()))
        hbraket = scipy.sparse.csc_matrix(scipy.sparse.linalg.expm_multiply(-ansatz.Full_JW_Ops[N-1]*parameters[N-1], hbraket))
        HAbra = hbraket.transpose()[0,:].conj()
        ket = hbraket.transpose()[1,:].transpose()
        hbra = hbraket.transpose()[2,:].conj()
    if len(ansatz.Full_JW_Ops)>0:
        cg[a][N] = 2*HAbra.dot(ansatz.Full_JW_Ops[N]).dot(ket).toarray()[0][0].real
    if N == 0:
        gradient.append(2*hbra.dot(ket).toarray()[0][0].real)
    if N<len(parameters)-1:
        N+=1
        cg, gradient = CG_Hessian(gradient, ansatz, ops, molecule, parameters, N, a, HAbra, hbra,  ket, cur_ket, cg)
    return cg, gradient

