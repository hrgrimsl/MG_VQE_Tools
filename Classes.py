import openfermion
import scipy
import numpy as np
import random

class Operator_Bank:
    def __init__(self, molecule, **kwargs):
         #Associate a Hamiltonian with this system
         self.hamiltonian = molecule.get_molecular_hamiltonian()
         self.two_index_hamiltonian = self.hamiltonian.one_body_tensor
         self.four_index_hamiltonian = self.hamiltonian.two_body_tensor
         self.JW_hamiltonian = openfermion.transforms.get_sparse_operator(self.hamiltonian)         

         #Construct reference ket
         self.HF_ket = scipy.sparse.csc_matrix(openfermion.jw_configuration_state(list(range(0, molecule.n_electrons)), molecule.n_qubits)).transpose()

         #Parse kwargs
         self.molecule = molecule
         self.screen_integrals = kwargs.get('screen_integrals', 'False')
         self.include_pqrs = kwargs.get('include_pqrs', 'False')
         self.screen_commutators = kwargs.get('screen_commutators', 'False')
         self.sort = kwargs.get('sort', None)
         self.spin_adapt = kwargs.get('spin_adapt', 'False')
 
         #Initialize op list
         self.SQ_Singles = []
         self.SQ_Doubles = []
         self.Singles = []
         self.Doubles = []
         self.JW_Singles = []
         self.JW_Doubles = []
         
         #Get unfiltered list
         if self.include_pqrs == 'True':
             self.PQRS()
         else:
             self.IJAB() 
        
         #Apply filters
         if self.screen_integrals == 'True':
             print('Screening by integrals...')
             self.Screen_Integrals()
         if self.screen_commutators == 'True':
             print('Screening by commutators with Hamiltonian (HF ansatz)...')
             self.Screen_Commutators()

         self.Full_Ops = self.Singles+self.Doubles
         self.Full_JW_Ops = self.JW_Singles+self.JW_Doubles
         
         #Spin adapt
         if self.spin_adapt == 'True':
             print('Spin adapting operators...')
             self.Spin_Adapt()
                
         #Apply sorting method
         if self.sort == None:
             print('Operators not sorted!')
             pass
         elif self.sort == 'commutators':
             print('Sorting operators by increasing commutators...')
             self.Sort_Commutators()
         else:
             print('Sorting operators by seed '+str(self.sort)+'...')
             self.Sort_Random(self.sort) 
    def __str__(self):
        return str(self.Full_Ops)

    def cache_ops(self):
        cache_string = str(len(self.Singles)+len(self.Doubles))+' operators: \n'
        for op in self.Full_Ops:
            cache_string+=str(op)+'\n'
        return cache_string

    def PQRS(self):
        
        #Singles
        for p in range(0, 2*self.molecule.n_orbitals):
            for q in range(p+1, 2*self.molecule.n_orbitals):
                if p%2 == q%2:
                    one_elec = openfermion.FermionOperator(((q,1),(p,0)))-openfermion.FermionOperator(((p,1),(q,0)))
                    self.SQ_Singles.append(one_elec)  
                    self.Singles.append(tuple([q,p]))
                    self.JW_Singles.append(openfermion.transforms.get_sparse_operator(one_elec, n_qubits = self.molecule.n_qubits))
                    
        #Doubles
        pairs = []
        for p in range(0, 2*self.molecule.n_orbitals):
            for q in range(p, 2*self.molecule.n_orbitals):
                pairs.append([p,q])
        for i in range(0, len(pairs)):
            for j in range(i+1, len(pairs)):
                p, r = pairs[j]
                q, s = pairs[i]
                if p%2+q%2 == r%2+s%2:
                    two_elec = openfermion.FermionOperator(((s,1),(r,1),(q,0),(p,0)))-openfermion.FermionOperator(((p,1),(q,1),(r,0),(s,0)))
                    self.SQ_Doubles.append(two_elec)
                    self.Doubles.append(tuple([s,r,q,p]))
                    self.JW_Doubles.append(openfermion.transforms.get_sparse_operator(two_elec, n_qubits = self.molecule.n_qubits))
                if tuple([q,r,s,p]) not in self.Doubles:
                    two_elec = openfermion.FermionOperator(((q,1),(r,1),(s,0),(p,0)))-openfermion.FermionOperator(((p,1),(s,1),(r,0),(q,0)))
                    self.SQ_Doubles.append(two_elec)
                    self.Doubles.append(tuple([q,r,s,p]))
                    self.JW_Doubles.append(openfermion.transforms.get_sparse_operator(two_elec, n_qubits = self.molecule.n_qubits))
                    
        assert(len(list(set(self.Doubles)))==len(self.Doubles))
    
    def IJAB(self):
        
        #Singles
        for i in range(0, self.molecule.n_electrons):
            for a in range(self.molecule.n_electrons, 2*self.molecule.n_orbitals):
                if a%2==i%2:
                    one_elec = openfermion.FermionOperator(((a,1),(i,0)))-openfermion.FermionOperator(((i,1),(a,0)))
                    self.SQ_Singles.append(one_elec)
                    self.Singles.append(tuple([a,i]))  
                    self.JW_Singles.append(openfermion.transforms.get_sparse_operator(one_elec, n_qubits = self.molecule.n_qubits))

        #Doubles
        for i in range(0, self.molecule.n_electrons):
            for j in range(i+1, self.molecule.n_electrons):
                for a in range(self.molecule.n_electrons, 2*self.molecule.n_orbitals):
                    for b in range(a+1, 2*self.molecule.n_orbitals):
                        if a%2+b%2==i%2+j%2:
                            two_elec = openfermion.FermionOperator(((b,1),(a,1),(j,0),(i,0)))-openfermion.FermionOperator(((i,1),(j,1),(a,0),(b,0)))
                            self.SQ_Doubles.append(two_elec)
                            self.Doubles.append(tuple([b,a,j,i]))
                            self.JW_Doubles.append(openfermion.transforms.get_sparse_operator(two_elec, n_qubits = self.molecule.n_qubits))

    def Screen_Integrals(self):

        #Singles
        New_Singles = []
        New_SQ_Singles = []
        New_JW_Singles = []
        for single in range(0, len(self.SQ_Singles)):
            a, i = self.Singles[single]
            if self.two_index_hamiltonian[a][i]!=0:
                New_Singles.append(self.Singles[single])
                New_SQ_Singles.append(self.SQ_Singles[single])
                New_JW_Singles.append(self.JW_Singles[single])
        self.Singles = New_Singles
        self.SQ_Singles = New_SQ_Singles
        self.JW_Singles = New_JW_Singles
  
        #Doubles
        New_Doubles = []
        New_SQ_Doubles = []
        New_JW_Doubles = []
        for double in range(0, len(self.SQ_Doubles)):
            b, a, j, i = self.Doubles[double]
            if self.four_index_hamiltonian[a][b][j][i] != 0:
                New_Doubles.append(self.Doubles[double])
                New_SQ_Doubles.append(self.SQ_Doubles[double])
                New_JW_Doubles.append(self.JW_Doubles[double])
        self.Doubles = New_Doubles
        self.SQ_Doubles = New_SQ_Doubles
        self.JW_Doubles = New_JW_Doubles

    def Screen_Commutators(self):

        #Singles
        New_Singles = []
        New_SQ_Singles = []
        New_JW_Singles = []
        for single in range(0, len(self.SQ_Singles)):
            if self.HF_ket.transpose().conj().dot(self.JW_hamiltonian).dot(self.JW_Singles[single]).dot(self.HF_ket).toarray()[0][0] != 0:
                New_Singles.append(self.Singles[single])
                New_SQ_Singles.append(self.SQ_Singles[single])
                New_JW_Singles.append(self.JW_Singles[single])
        self.Singles = New_Singles
        self.SQ_Singles = New_SQ_Singles
        self.JW_Singles = New_JW_Singles
  
        #Doubles
        New_Doubles = []
        New_SQ_Doubles = []
        New_JW_Doubles = []
        for double in range(0, len(self.SQ_Doubles)):
            if self.HF_ket.transpose().conj().dot(self.JW_hamiltonian).dot(self.JW_Doubles[double]).dot(self.HF_ket).toarray()[0][0] != 0:
                New_Doubles.append(self.Doubles[double])
                New_SQ_Doubles.append(self.SQ_Doubles[double])
                New_JW_Doubles.append(self.JW_Doubles[double])
        self.Doubles = New_Doubles
        self.SQ_Doubles = New_SQ_Doubles
        self.JW_Doubles = New_JW_Doubles

    def Spin_Adapt(self):
        New_JW_Ops = []
        New_Ops = []
        #Doubles
        for chi_a in range(0, self.molecule.n_electrons):
            for chi_b in range(chi_a+1, self.molecule.n_electrons):
                for chi_i in range(0, self.molecule.n_electrons):
                    for chi_j in range(chi_i+1, self.molecule.n_electrons):
                        spatial_transition = [set([chi_a, chi_b]), set([chi_i, chi_j])]
                        new_op = None
                        for op in range(0, len(self.Full_JW_Ops)):
                            if len(self.Full_Ops[op]) == 4:
                                op_tran = [set([int(round(self.Full_Ops[op][0]/2)+1),int(round(self.Full_Ops[op][1]/2)+1)]),set([int(round(self.Full_Ops[op][2]/2)+1),int(round(self.Full_Ops[op][3]/2)+1)])]
                                if op_tran == spatial_transition:
                                    if new_op != None:
                                        new_op += self.Full_JW_Ops[op]
                                    else:
                                        new_op = self.Full_JW_Ops[op]
                        if new_op!= None:
                           New_Ops.append(spatial_transition)
                           New_JW_Ops.append(new_op)         
        #Singles
        for chi_a in range(0, self.molecule.n_electrons):
            for chi_b in range(chi_a, self.molecule.n_electrons):
                spatial_transition = [set([chi_a, chi_b])]
                new_op = None
                for op in range(0, len(self.Full_JW_Ops)):
                    if len(self.Full_Ops[op]) == 2:
                        op_tran = [set([int(round(self.Full_Ops[op][0]/2)+1),int(round(self.Full_Ops[op][1]/2)+1)])]
                        if op_tran == spatial_transition:
                            if new_op != None:
                                new_op += self.Full_JW_Ops[op]
                            else:
                                new_op = self.Full_JW_Ops[op]
                if new_op!= None:
                    New_Ops.append(spatial_transition)
                    New_JW_Ops.append(new_op)         
        self.Full_JW_Ops = New_JW_Ops
        self.Full_Ops = New_Ops
 
    def Sort_Commutators(self):
       comms = []
       for op in self.Full_JW_Ops:
           comms.append(self.HF_ket.transpose().conj().dot(self.JW_hamiltonian).dot(op).dot(self.HF_ket).toarray()[0][0].real)
       indices = np.unravel_index(np.argsort(comms, axis=None), np.array(comms).shape)
       self.Full_JW_Ops = list(np.array(self.Full_JW_Ops)[indices])
       self.Full_Ops = list(np.array(self.Full_Ops)[indices])

    def Sort_Random(self, seed):
       random.seed(int(seed))
       ops = list(zip(self.Full_JW_Ops, self.Full_Ops))
       random.shuffle(ops)
       self.Full_JW_Ops, self.Full_Ops = zip(*ops)

class Ansatz_Operations:
    def __init__(self, ops):
        self.Full_JW_Ops = []
        self.Full_Ops = []
        self.HF_ket = ops.HF_ket
        self.molecule = ops.molecule
        self.JW_hamiltonian = ops.JW_hamiltonian
    def __str__(self):
        return(str(self.Full_Ops))
