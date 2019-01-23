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
    if algorithm == 'Force':
        outcome = Force(molecule, ops, theta_tightness, ADAPT_tightness, logging)

    return outcome
def Callback(optimization):
    print('...')
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
        try:
            logging.info(energy)
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
    #A, B in pool
    hessian['ce'] = scipy_hessians[-1]
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
        #grad = gradient
        max = 0
        num = None
        for i in range(0, len(grad)):
            delta_e = 0
            for j in range(0, len(gradient)):
                delta_e-=.5*grad[i]*hinv[i][j]*gradient[j]
                #delta_e-=.5*grad[i]*hinv[i][j]*grad[j]
                if j>len(ops.Full_JW_Ops) and gradient[j]!=0:
                    print(j)
                    print(gradient[j])
            if abs(delta_e)>max:
                num = i
                max = abs(delta_e)
        ansatz.Full_JW_Ops.insert(0, ops.Full_JW_Ops[num])
        ansatz.Full_Ops.insert(0, ops.Full_Ops[num])
        parameters = list(parameters)
        parameters.insert(0, 0)
        print('\nIteration '+str(len(parameters))+'.\n')
        print('Next operation: {:10s}'.format(str(ops.Full_Ops[num])))
        print('Predicted energy change: {:10.14f}'.format(-max))
        OptRes = VQE(molecule, parameters, ansatz, theta_tightness, logging)
        parameters = OptRes.x
        print('Energy change = {:+10.14}'.format(OptRes.fun-energy))
        energy = OptRes.fun
        scipy_hessians.append(np.linalg.pinv(OptRes.hess_inv))


        
    return OptRes
        
    #while hessian_norm == None or hessian_norm > ADAPT_tightness:


def Hess(ansatz, ops, molecule, parameters):
    parameters = list(parameters)
    parameters = [0 for i in range(0, len(ops.Full_JW_Ops))]+parameters
    hess = []
    for i in range(0, len(parameters)):
        hess.append([])
        hess[-1] = list(scipy.optimize.approx_fprime(parameters, Grad, 1e-6, ansatz, ops, molecule, i))
    return np.array(hess)

def Grad(parameters, ansatz, ops, molecule, term):
    grad =  scipy.optimize.approx_fprime(parameters, Energy, 1e-6, ansatz, ops, molecule)
    return grad[term]

def Energy(parameters, ansatz, ops, molecule):
    ket = ops.HF_ket
    Ops = ops.Full_JW_Ops+ansatz.Full_JW_Ops
    for op in reversed(range(0, len(Ops))):
        ket = scipy.sparse.linalg.expm_multiply(Ops[op]*parameters[op], ket)
    return ket.transpose().conj().dot(ops.JW_hamiltonian).dot(ket).toarray()[0][0].real
    




