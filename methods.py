from evaluations import *

def Optimize(molecule, ops, logging, **kwargs):
    algorithm = kwargs.get('algorithm', 'VQE')
    theta_tightness = float(kwargs.get('theta_tightness', '1e-5'))
    STAG_tightness = float(kwargs.get('theta_tightness', '1e-5'))
    if algorithm == 'VQE':
        outcome = VQE(molecule, ops, theta_tightness)
        return outcome

def VQE(molecule, ops, theta_tightness):
    #Initialize parameters
    parameters = []
    for i in range(0, len(ops.Full_Ops)):
        parameters.append(0)
    print('Performing optimization of parameters...')
    optimization = scipy.optimize.minimize(Trotter_SPE, parameters, jac = Trotter_Gradient, args = (ops), method = 'BFGS', options = {'gtol': float(theta_tightness), 'disp': False})
    print('Parameters optimized in '+str(optimization.nit)+' iterations!')
    return optimization
