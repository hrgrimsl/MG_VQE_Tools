import openfermion
from math import floor
import scipy
import numpy as np
import random
import copy
import math

def Make_S2(n_orb):
# {{{
    ap =scipy.sparse.csc_matrix(np.array([[0, 0], [1, 0]]))    #creation operator
    am =scipy.sparse.csc_matrix( np.array([[0, 1], [0, 0]])) #annihilation operator
    no =scipy.sparse.csc_matrix( np.array([[0, 0], [0, 1]]))     #number operator
    ho =scipy.sparse.csc_matrix( np.array([[1, 0], [0, 0]]))     #hole operator
    I2 =scipy.sparse.csc_matrix( np.array([[1, 0], [0, 1]]))     #identity operator
    Iz =scipy.sparse.csc_matrix( np.array([[1, 0], [0, -1]]))    #pauli z operat
    S2 =scipy.sparse.csc_matrix( np.zeros((4**n_orb,4**n_orb)))
    s2 =scipy.sparse.csc_matrix( np.array([[0,0],[0,0.75]]))

    for i in range(0,n_orb):
        bfor  = 2*i
        aftr  = 2*n_orb-2*i-2

        Ia = np.eye(np.power(2,bfor))
        Ib = np.eye(np.power(2,aftr))
        a_temp = scipy.sparse.kron(s2,I2)
        b_temp = scipy.sparse.kron(I2,s2)
        S2a = scipy.sparse.kron(Ia,scipy.sparse.kron(a_temp,Ib))
        S2b = scipy.sparse.kron(Ia,scipy.sparse.kron(b_temp,Ib))

        S2 += abs(S2a -S2b)
        

        for j in range(i+1,n_orb):
            
            intr = 2*j-2*i-2 
            aftr = 2*n_orb-2*j-2

            Ib = np.eye(np.power(2,intr))
            Zb = np.eye(1)
            for k in range(2*i+2,2*j):
                Zb = scipy.sparse.kron(Zb,Iz)

            Ic = np.eye(np.power(2,aftr))
            Zc = np.eye(1)
            for k in range(2*j,2*n_orb-2):
                Zc = scipy.sparse.kron(Zc,Iz)
            
            assert(Zc.shape == Ic.shape)
            assert(Zb.shape == Ib.shape)

            
            Sptemp = scipy.sparse.kron(ap,am) 
            Smtemp = scipy.sparse.kron(am,ap) 
            ANtemp = scipy.sparse.kron(no,I2)
            BNtemp = scipy.sparse.kron(I2,no)

            ##CASE A
            aiaj = scipy.sparse.kron(Ia,scipy.sparse.kron(Sptemp,scipy.sparse.kron(Ib,scipy.sparse.kron(Smtemp,Ic))))
            S2  +=  (aiaj)

            ##CASE B
            aiaj = scipy.sparse.kron(Ia,scipy.sparse.kron(Smtemp,scipy.sparse.kron(Ib,scipy.sparse.kron(Sptemp,Ic))))
            S2  +=  (aiaj)

            ##CASE C
            aiaj = scipy.sparse.kron(Ia,scipy.sparse.kron(ANtemp,scipy.sparse.kron(Ib,scipy.sparse.kron(BNtemp,Ic))))
            S2  -= 0.5 * (aiaj)

            ##CASE D
            aiaj = scipy.sparse.kron(Ia,scipy.sparse.kron(BNtemp,scipy.sparse.kron(Ib,scipy.sparse.kron(ANtemp,Ic))))
            S2  -= 0.5 * (aiaj)

            ##CASE E
            aiaj = scipy.sparse.kron(Ia,scipy.sparse.kron(ANtemp,scipy.sparse.kron(Ib,scipy.sparse.kron(ANtemp,Ic))))
            S2  += 0.5 * (aiaj)

            ##CASE F
            aiaj = scipy.sparse.kron(Ia,scipy.sparse.kron(BNtemp,scipy.sparse.kron(Ib,scipy.sparse.kron(BNtemp,Ic))))
            S2  += 0.5 * (aiaj)

    return scipy.sparse.csc_matrix(S2)
    # }}}



       

class Operator_Bank:
    def __init__(self, molecule, **kwargs):
         self.ecp = 0
         #We give doccs and noccs as spatial orbitals, soccs as spin-orbitals 
         self.active = (kwargs.get('active', 'False'))
         self.intra = (kwargs.get('intra', None))
         self.inter = (kwargs.get('inter', None))
         self.flip = (kwargs.get('flip','False'))
         
         self.active_doccs = (kwargs.get('occs', 'False'))
         self.soccs = (kwargs.get('soccs', 'False'))
         self.repeats = (kwargs.get('repeats', 'True'))
         self.kup = (kwargs.get('kup', 'False'))
         doccs = []
         if self.soccs == 'False':
             self.soccs = []
         else:
             self.soccs = [int(s) for s in self.soccs.split(',')]
                
         self.molecule = molecule

         if self.active == 'False':
             occupation = []
             active_indices = [i for i in range(0, molecule.n_orbitals)]
             unpaired = molecule.multiplicity-1
             occs = [i for i in range(0, int(((molecule.n_electrons-unpaired)/2)))]
             if occs == []:
                 soccs = [i for i in range(0, unpaired+1)]
             else:    
                 soccs = [i for i in range(occs[-1]+1, occs[-1]+unpaired+1)]
             noccs = sorted(list(set(active_indices)-set(occs)-set(occupation)-set(soccs))) 

             try:
                 self.S2 = Make_S2(molecule.n_orbitals) 
             except:
                 print('S2 calculation unsuccessful or not attempted.') 

         else:
             try: 
                 self.S2 = Make_S2(molecule.n_orbitals) 
             except: 
                 pass

         doccs = [i for i in range(0, self.molecule.n_fdoccs)]
         if self.molecule.active!=None:
             active = [i for i in range(len(doccs), len(doccs)+len(self.molecule.active))]
         else:
             active = [i for i in range(0, self.molecule.n_orbitals)]

         self.hamiltonian = molecule.get_molecular_hamiltonian(occupied_indices = doccs, active_indices = active) 
         self.ecp = 0 
         self.two_index_hamiltonian = self.hamiltonian.one_body_tensor
         self.four_index_hamiltonian = self.hamiltonian.two_body_tensor
         self.JW_hamiltonian = openfermion.transforms.get_sparse_operator(self.hamiltonian)         
         #Construct spaces
         self.aoccs = [i for i in range(0, self.molecule.n_electrons) if i%2 == 0 and i>=2*int(self.ecp)]
         self.anoccs = [i for i in range(self.molecule.n_electrons, self.molecule.n_orbitals*2) if i%2 == 0]
         self.boccs = [i for i in range(0, self.molecule.n_electrons) if i%2 == 1 and i>=2*int(self.ecp)]
         self.bnoccs = [i for i in range(self.molecule.n_electrons, self.molecule.n_orbitals*2) if i%2 == 1]
         self.alphas = self.aoccs+self.anoccs
         self.betas = self.boccs+self.bnoccs
         #Construct reference ket
         occ = []

         spinorbitals = self.molecule.n_electrons-2*self.molecule.n_fdoccs
         soccs = int(self.molecule.multiplicity-1)

         try:
             doccs = int((self.molecule.n_electrons-soccs)/2)

         except:
             doccs = int((self.molecule.n_electrons-soccs)/2)
         for i in range(0, doccs):
             occ.append(2*i)
             occ.append(2*i+1)
         for i in range(doccs, doccs+soccs):
             occ.append(2*i)
         

         if self.molecule.occ!='None' and self.molecule.occ!=None:
             occ = self.molecule.occ
         
         print('Electron configuration: '+str(occ))


         self.HF_ket = scipy.sparse.csc_matrix(openfermion.jw_configuration_state(occ, molecule.n_qubits)).transpose()
         
         #Parse kwargs

         self.include_pqrs = kwargs.get('include_pqrs', 'False')
         self.screen_commutators = kwargs.get('screen_commutators', 'False')
         self.sort = kwargs.get('sort', None)
         self.spin_adapt = kwargs.get('spin_adapt', 'False')
         self.gs = kwargs.get('gs', 'False')
 
         #Initialize op list
         self.SQ_Singles = []
         self.SQ_Doubles = []
         self.Singles = []
         self.Doubles = []
         self.Full_JW_Ops = []
         
         if self.gs == 'True':
             self.GS()
                              
         elif self.include_pqrs == 'True':
             if self.spin_adapt == 'False':
                 self.PQRS()
             else:
                 if self.molecule.multiplicity == 1:
                     self.GSD_Singlet()
                 else:
                     self.GSD_Triplet()
         else:
             if self.spin_adapt == 'False':
                 self.IJAB() 
             else:
                 if self.molecule.multiplicity == 1:
                     self.SD_Singlet()
                 else:
                     self.SD_Triplet()



         if self.flip == 'False':
             self.Full_Ops = self.Singles+self.Doubles
             self.Full_SQ_Ops = self.SQ_Singles+self.SQ_Doubles
         else:
             self.Full_Ops = self.Doubles+self.Singles
             self.Full_SQ_Ops = self.SQ_Doubles+self.SQ_Singles
         if self.inter!=None:
              random.seed(int(self.inter))
              c = list(zip(self.Full_SQ_Ops, self.Full_Ops))
              random.shuffle(c)
              self.Full_SQ_Ops, self.Full_Ops = zip(*c)         
         for op in self.Full_SQ_Ops:

             op = openfermion.normal_ordered(op)

             if op.many_body_order()>0:
                 self.Full_JW_Ops.append(openfermion.transforms.get_sparse_operator(op, n_qubits = self.molecule.n_qubits))
         #Apply filters
         if self.screen_commutators == 'True':

             self.Screen_Commutators()

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
        for i in range(0, self.molecule.n_orbitals*2):
             for a in range(0, self.molecule.n_orbitals*2):
                 if a<i and a%2==i%2:
                    one_elec = openfermion.FermionOperator(((a,1),(i,0)))-openfermion.FermionOperator(((i,1),(a,0)))
                    norm = 0
                    for term in one_elec.terms:
                        norm += one_elec.terms[term]*one_elec.terms[term]
                    self.SQ_Singles.append(one_elec/np.sqrt(norm))
                    self.Singles.append([a,i])

    
        #Doubles
        pairs = []
        for i in range(0, self.molecule.n_orbitals*2):
            for j in range(i+1, self.molecule.n_orbitals*2):
                pairs.append([i,j])
        for p in range(0, len(pairs)):
            for q in range(p+1, len(pairs)):
                j,i = pairs[p]
                b,a = pairs[q]
                if i%2+j%2==a%2+b%2 and (a!=i or b!=j):
                    norm = 0
                    two_elec = openfermion.FermionOperator(((a,1),(b,1),(i,0),(j,0)))-openfermion.FermionOperator(((j,1),(i,1),(b,0),(a,0)))
                    for term in two_elec.terms:
                        norm += two_elec.terms[term]*two_elec.terms[term]
                    self.SQ_Doubles.append(two_elec/np.sqrt(norm))
                    self.Doubles.append([a,i,b,j])

    def GSD_Singlet(self):
        #Singles
        num = openfermion.FermionOperator(())-openfermion.FermionOperator(())
        for i in range(int(self.ecp), self.molecule.n_orbitals):
            for a in range(i+1, self.molecule.n_orbitals):
                one_elec = (1/np.sqrt(2))*openfermion.FermionOperator(((2*a-2*self.ecp,1),(2*i-2*self.ecp,0)))
                one_elec += (1/np.sqrt(2))*openfermion.FermionOperator(((2*a+1-2*self.ecp,1),(2*i+1-2*self.ecp,0)))
                one_elec -= openfermion.hermitian_conjugated(one_elec)
                one_elec = openfermion.normal_ordered(one_elec)
                norm = 0
                for term in one_elec.terms:
                    norm += one_elec.terms[term]*one_elec.terms[term]
                if one_elec.many_body_order()>0 and norm!=0:
                    self.SQ_Singles.append(one_elec/np.sqrt(norm))
                    self.Singles.append([a-self.ecp,i-self.ecp])               
            num+=openfermion.FermionOperator(((i*2,1),(i*2,0)))
            num+=openfermion.FermionOperator(((i*2+1,1),(i*2+1,0)))

        #Doubles
        if self.kup == 'False':
            pairs = []
            for j in range(int(self.ecp), self.molecule.n_orbitals):
                for i in range(j, self.molecule.n_orbitals):
                    pairs.append([i-self.ecp,j-self.ecp])
            for pair in range(0, len(pairs)):
                for pair2 in range(pair+1, len(pairs)):
                    i, j = pairs[pair]
                    a, b = pairs[pair2]           
                    two_elec = 2*openfermion.FermionOperator(((2*a,1),   (2*b,1),     (2*i,0),    (2*j,0)))
                    two_elec += 2*openfermion.FermionOperator(((2*a+1,1),(2*b+1,1),   (2*i+1,0),  (2*j+1,0)))
                    two_elec += openfermion.FermionOperator(((2*a+1,1),  (2*b,1),     (2*i+1,0),  (2*j,0)))
                    two_elec += openfermion.FermionOperator(((2*a,1),    (2*b+1,1),   (2*i,0),    (2*j+1,0)))
                    two_elec += openfermion.FermionOperator(((2*a,1),  (2*b+1,1),     (2*i+1,0),  (2*j,0)))
                    two_elec += openfermion.FermionOperator(((2*a+1,1),    (2*b,1),   (2*i,0),    (2*j+1,0)))
                    norm = 0
                    two_elec -= openfermion.hermitian_conjugated(two_elec)
                    two_elec = openfermion.normal_ordered(two_elec)
                    for term in two_elec.terms:
                        norm += two_elec.terms[term]*two_elec.terms[term]
                    if two_elec.many_body_order()>0 and norm!=0:
                        self.SQ_Doubles.append(two_elec/np.sqrt(norm))
                        self.Doubles.append([a,b,i,j])               
                    
                     
                    two_elec = openfermion.FermionOperator(((2*a+1,1),(2*b,1),(2*i+1,0),(2*j,0)))
                    two_elec += openfermion.FermionOperator(((2*a,1),(2*b+1,1),(2*i,0),(2*j+1,0)))
                    two_elec -= openfermion.FermionOperator(((2*a,1),(2*b+1,1),(2*i+1,0),(2*j,0)))
                    two_elec -= openfermion.FermionOperator(((2*a+1,1),(2*b,1),(2*i,0),(2*j+1,0)))
                    norm = 0
                    two_elec -= openfermion.hermitian_conjugated(two_elec)
                    two_elec = openfermion.normal_ordered(two_elec)
                    for term in two_elec.terms:
                        norm += two_elec.terms[term]*two_elec.terms[term]

                    if two_elec.many_body_order()>0 and norm!=0:
                        self.SQ_Doubles.append(two_elec/np.sqrt(norm))
                        self.Doubles.append([a,b,i,j])

        else:
             pairs = []
             for j in range(0, self.molecule.n_orbitals):
                 for i in range(j, self.molecule.n_orbitals):
                     if i!=j:
                         two_elec = openfermion.FermionOperator(((2*j,1),(2*j+1,1),(2*i,0),(2*i+1,0)))
                         two_elec -= openfermion.hermitian_conjugated(two_elec)
                         two_elec = openfermion.normal_ordered(two_elec)
                         for term in two_elec.terms:
                             norm += two_elec.terms[term]*two_elec.terms[term]
                         if two_elec.many_body_order()>0 and norm!=0:
                             self.SQ_Doubles.append(two_elec/np.sqrt(norm))
                             self.Doubles.append([j,j,i,i])
                   
    def GS(self):
        #Singles
        for i in range(int(self.ecp), self.molecule.n_orbitals):
            for a in range(i+1, self.molecule.n_orbitals):
                one_elec = (1/np.sqrt(2))*openfermion.FermionOperator(((2*a-2*self.ecp,1),(2*i-2*self.ecp,0)))
                one_elec += (1/np.sqrt(2))*openfermion.FermionOperator(((2*a+1-2*self.ecp,1),(2*i+1-2*self.ecp,0)))
                one_elec -= openfermion.hermitian_conjugated(one_elec)
                one_elec = openfermion.normal_ordered(one_elec)
                norm = 0
                for term in one_elec.terms:
                    norm += one_elec.terms[term]*one_elec.terms[term]
                if one_elec.many_body_order()>0 and norm!=0:
                    self.SQ_Singles.append(one_elec/np.sqrt(norm))
                    self.Singles.append([a-self.ecp,i-self.ecp])               
         
        

    def GSD_Triplet(self):
        #Singles
        for i in range(int(self.ecp), self.molecule.n_orbitals):
            for a in range(i+1, self.molecule.n_orbitals):
                one_elec = (1/np.sqrt(2))*openfermion.FermionOperator(((2*a-2*self.ecp,1),(2*i-2*self.ecp,0)))
                one_elec += (1/np.sqrt(2))*openfermion.FermionOperator(((2*a+1-2*self.ecp,1),(2*i+1-2*self.ecp,0)))
                one_elec -= openfermion.hermitian_conjugated(one_elec)
                one_elec = openfermion.normal_ordered(one_elec)
                norm = 0
                for term in one_elec.terms:
                    norm += one_elec.terms[term]*one_elec.terms[term]
                if one_elec.many_body_order()>0 and norm!=0:
                    self.SQ_Singles.append(one_elec/np.sqrt(norm))
                    self.Singles.append([a-self.ecp,i-self.ecp])               
                 
        #Doubles
        pairs = []
        for j in range(0, self.molecule.n_orbitals):
            for i in range(j, self.molecule.n_orbitals):
                pairs.append([i,j])
        for pair in range(0, len(pairs)):
            for pair2 in range(pair+1, len(pairs)):
                i, j = pairs[pair]
                a, b = pairs[pair2]           
                two_elec = 2*openfermion.FermionOperator(((2*a,1),   (2*b,1),     (2*i,0),    (2*j,0)))
                two_elec += 2*openfermion.FermionOperator(((2*a+1,1),(2*b+1,1),   (2*i+1,0),  (2*j+1,0)))
                two_elec += openfermion.FermionOperator(((2*a+1,1),  (2*b,1),     (2*i+1,0),  (2*j,0)))
                two_elec += openfermion.FermionOperator(((2*a,1),    (2*b+1,1),   (2*i,0),    (2*j+1,0)))
                two_elec += openfermion.FermionOperator(((2*a,1),  (2*b+1,1),     (2*i+1,0),  (2*j,0)))
                two_elec += openfermion.FermionOperator(((2*a+1,1),    (2*b,1),   (2*i,0),    (2*j+1,0)))
                norm = 0
                two_elec -= openfermion.hermitian_conjugated(two_elec)
                two_elec = openfermion.normal_ordered(two_elec)
                for term in two_elec.terms:
                    norm += two_elec.terms[term]*two_elec.terms[term]
                if two_elec.many_body_order()>0 and norm!=0:
                    self.SQ_Doubles.append(two_elec/np.sqrt(norm))
                    self.Doubles.append([a,b,i,j])               

                two_elec = openfermion.FermionOperator(((2*a+1,1),(2*b,1),(2*i+1,0),(2*j,0)))
                two_elec += openfermion.FermionOperator(((2*a,1),(2*b+1,1),(2*i,0),(2*j+1,0)))
                two_elec -= openfermion.FermionOperator(((2*a,1),(2*b+1,1),(2*i+1,0),(2*j,0)))
                two_elec -= openfermion.FermionOperator(((2*a+1,1),(2*b,1),(2*i,0),(2*j+1,0)))
                norm = 0
                two_elec -= openfermion.hermitian_conjugated(two_elec)
                two_elec = openfermion.normal_ordered(two_elec)
                for term in two_elec.terms:
                    norm += two_elec.terms[term]*two_elec.terms[term]
                if two_elec.many_body_order()>0 and norm!=0:
                    self.SQ_Doubles.append(two_elec/np.sqrt(norm))
                    self.Doubles.append([a,b,i,j])

    def GSD_Singlet_2(self):
        #Now with triples!:
        #Singles
        for i in range(int(self.ecp), self.molecule.n_orbitals):
            for a in range(i+1, self.molecule.n_orbitals):
                one_elec = (1/np.sqrt(2))*openfermion.FermionOperator(((2*a-2*self.ecp,1),(2*i-2*self.ecp,0)))
                one_elec_2 = (1/np.sqrt(2))*openfermion.FermionOperator(((2*a+1-2*self.ecp,1),(2*i+1-2*self.ecp,0)))

                one_elec -= openfermion.hermitian_conjugated(one_elec)
                one_elec_2 -= openfermion.hermitian_conjugated(one_elec_2)
                one_elec = one_elec*one_elec_2-one_elec_2*one_elec
                one_elec = openfermion.normal_ordered(one_elec)

                norm = 0
                for term in one_elec.terms:
                    norm += one_elec.terms[term]*one_elec.terms[term]
                if one_elec.many_body_order()>0 and norm!=0:
                    self.SQ_Singles.append(one_elec/np.sqrt(norm))
                    self.Singles.append([a-self.ecp,i-self.ecp])               
                 
        #Doubles
        pairs = []
        for j in range(int(self.ecp), self.molecule.n_orbitals):
            for i in range(j, self.molecule.n_orbitals):
                pairs.append([i-self.ecp,j-self.ecp])
        for pair in range(0, len(pairs)):
            for pair2 in range(pair+1, len(pairs)):
                i, j = pairs[pair]
                a, b = pairs[pair2]
                if i == b:
                    continue
                two_elec = 2*openfermion.FermionOperator(((2*a,1),   (2*b,1),     (2*i,0),    (2*j,0)))
                two_elec2 = 2*openfermion.FermionOperator(((2*a+1,1),(2*b+1,1),   (2*i+1,0),  (2*j+1,0)))
                two_elec3 = openfermion.FermionOperator(((2*a+1,1),  (2*b,1),     (2*i+1,0),  (2*j,0)))
                two_elec4 = openfermion.FermionOperator(((2*a,1),    (2*b+1,1),   (2*i,0),    (2*j+1,0)))
                two_elec5 = openfermion.FermionOperator(((2*a,1),  (2*b+1,1),     (2*i+1,0),  (2*j,0)))
                two_elec6 = openfermion.FermionOperator(((2*a+1,1),    (2*b,1),   (2*i,0),    (2*j+1,0)))
                norm = 0
                two_elec-= openfermion.hermitian_conjugated(two_elec)
                two_elec2-= openfermion.hermitian_conjugated(two_elec2)
                two_elec3-= openfermion.hermitian_conjugated(two_elec3)
                two_elec4-= openfermion.hermitian_conjugated(two_elec4)
                two_elec5-= openfermion.hermitian_conjugated(two_elec5)
                two_elec6-= openfermion.hermitian_conjugated(two_elec6)

                

                #print(openfermion.normal_ordered(two_elec*two_elec3-two_elec3*two_elec))
                #print(openfermion.normal_ordered(two_elec*two_elec4-two_elec4*two_elec))
                #print(openfermion.normal_ordered(two_elec*two_elec5-two_elec5*two_elec))
                #print(openfermion.normal_ordered(two_elec*two_elec6-two_elec6*two_elec))
                #print(openfermion.normal_ordered(two_elec2*two_elec3-two_elec3*two_elec2))
                #print(openfermion.normal_ordered(two_elec2*two_elec4-two_elec4*two_elec2))
                #print(openfermion.normal_ordered(two_elec2*two_elec5-two_elec5*two_elec2))
                #print(openfermion.normal_ordered(two_elec2*two_elec6-two_elec6*two_elec2))
                #print(openfermion.normal_ordered(two_elec3*two_elec4-two_elec4*two_elec3))
                #print(openfermion.normal_ordered(two_elec3*two_elec5-two_elec5*two_elec3))
                #print(openfermion.normal_ordered(two_elec3*two_elec6-two_elec6*two_elec3))
                #print(openfermion.normal_ordered(two_elec4*two_elec5-two_elec5*two_elec4))
                #print(openfermion.normal_ordered(two_elec4*two_elec6-two_elec6*two_elec4))
                #print(openfermion.normal_ordered(two_elec5*two_elec6-two_elec6*two_elec5))
                for term in two_elec.terms:
                    norm += two_elec.terms[term]*two_elec.terms[term]
                if two_elec.many_body_order()>0 and norm!=0:
                    self.SQ_Doubles.append(two_elec/np.sqrt(norm))
                    self.Doubles.append([a,b,i,j])               

                two_elec = openfermion.FermionOperator(((2*a+1,1),(2*b,1),(2*i+1,0),(2*j,0)))
                two_elec += openfermion.FermionOperator(((2*a,1),(2*b+1,1),(2*i,0),(2*j+1,0)))
                two_elec -= openfermion.FermionOperator(((2*a,1),(2*b+1,1),(2*i+1,0),(2*j,0)))
                two_elec -= openfermion.FermionOperator(((2*a+1,1),(2*b,1),(2*i,0),(2*j+1,0)))
                norm = 0
                two_elec -= openfermion.hermitian_conjugated(two_elec)
                two_elec = openfermion.normal_ordered(two_elec)
                for term in two_elec.terms:
                    norm += two_elec.terms[term]*two_elec.terms[term]
                if two_elec.many_body_order()>0 and norm!=0:
                    self.SQ_Doubles.append(two_elec/np.sqrt(norm))
                    self.Doubles.append([a,b,i,j])
        exit()
        '''
        #Triples
        pairs = []
        for j in range(int(self.ecp), self.molecule.n_qubits):
            for i in range(j+1, self.molecule.n_qubits):
                for k in range(j+1, self.molecule.n_qubits):
                    pairs.append([i-self.ecp,j-self.ecp,k-self.ecp])
        for pair in range(0, len(pairs)):
            for pair2 in range(pair+1, len(pairs)):
                i, j,k = pairs[pair]
                a, b,c = pairs[pair2]           
                three_elec = openfermion.FermionOperator(((a,1),   (b,1), (c,1),(k,0)  ,(i,0),    (j,0)))
                norm = 0
                three_elec -= openfermion.hermitian_conjugated(three_elec)
                three_elec = openfermion.normal_ordered(three_elec)
                for term in three_elec.terms:
                    norm += three_elec.terms[term]*three_elec.terms[term]
                if three_elec.many_body_order()>0 and norm!=0:
                    self.SQ_Doubles.append(three_elec/np.sqrt(norm))
                    self.Doubles.append([a,b,c,k,i,j])               
         '''

    def IJAB(self):
        
        #Singles
        #aa
        for i in self.aoccs:
            for a in self.anoccs:
                one_elec = 2**-.5*openfermion.FermionOperator(((a-self.ecp*2,1),(i-self.ecp*2,0)))-2**-.5*openfermion.FermionOperator(((i-self.ecp*2,1),(a-self.ecp*2,0)))
                self.SQ_Singles.append(one_elec)
                self.Singles.append([a-self.ecp*2,i-self.ecp*2])

        #bb
        for i in self.boccs:
            for a in self.bnoccs:
                one_elec = 2**-.5*openfermion.FermionOperator(((a-self.ecp*2,1),(i-self.ecp*2,0)))-2**-.5*openfermion.FermionOperator(((i-self.ecp,1),(a-self.ecp,0)))
                self.SQ_Singles.append(one_elec)
                self.Singles.append([a-self.ecp,i-self.ecp])

        #Doubles
        occs = []
        noccs = []

        for i in range(0, self.molecule.n_electrons):
            for j in range(i+1, self.molecule.n_electrons):
                occs.append([i-self.ecp*2,j-self.ecp*2])
        for a in range(self.molecule.n_electrons, self.molecule.n_orbitals*2):
            for b in range(a+1, self.molecule.n_orbitals*2):
                noccs.append([a-self.ecp*2,b-self.ecp*2])

        for occ in occs:
            for nocc in noccs:
                j, i = occ
                b, a = nocc
                if a%2+b%2==i%2+j%2:
                    two_elec = 2**-.5*openfermion.FermionOperator(((a,1),(b,1),(i,0),(j,0)))-2**-.5*openfermion.FermionOperator(((j,1),(i,1),(b,0),(a,0)))
                    self.Doubles.append([a,i,b,j])
                    self.SQ_Doubles.append(two_elec)

    def SD_Singlet(self):
         #Singles
         for i in range(int(self.ecp), int(self.molecule.n_electrons/2)):
             for a in range(int(self.molecule.n_electrons/2), self.molecule.n_orbitals):
                 one_elec = (1/np.sqrt(2))*openfermion.FermionOperator(((2*a-self.ecp*2,1),(2*i-self.ecp*2,0)))
                 one_elec += (1/np.sqrt(2))*openfermion.FermionOperator(((2*a+1-self.ecp*2,1),(2*i+1-self.ecp*2,0)))
                 one_elec -= openfermion.hermitian_conjugated(one_elec)
                 one_elec = openfermion.normal_ordered(one_elec)
                 norm = 0
                 for term in one_elec.terms:
                     norm += one_elec.terms[term]*one_elec.terms[term]
                 if one_elec.many_body_order()>0 and norm!=0:

                     self.SQ_Singles.append(one_elec/np.sqrt(norm))
                     self.Singles.append([a-self.ecp,i-self.ecp])               
         
         if self.intra!=None:
              print('Intra is '+str(self.intra))
              random.seed(int(self.intra))
              c = list(zip(self.SQ_Singles, self.Singles))
              random.shuffle(c)
              self.SQ_Singles, self.Singles = zip(*c)
 
              
         #Doubles
         pairs = []
         for j in range(int(self.ecp), int(self.molecule.n_electrons/2)):
             for i in range(j, int(self.molecule.n_electrons/2)):
                 for b in range(int(self.molecule.n_electrons/2), self.molecule.n_orbitals):
                     for a in range(b, self.molecule.n_orbitals):
                          two_elec = 2*openfermion.FermionOperator(((2*a-self.ecp*2,1),   (2*b-self.ecp*2,1),     (2*i-self.ecp*2,0),    (2*j-self.ecp*2,0)))
                          two_elec += 2*openfermion.FermionOperator(((2*a+1-self.ecp*2,1),(2*b+1-self.ecp*2,1),   (2*i+1-self.ecp*2,0),  (2*j+1-self.ecp*2,0)))
                          two_elec += openfermion.FermionOperator(((2*a+1-self.ecp*2,1),  (2*b-self.ecp*2,1),     (2*i+1-self.ecp*2,0),  (2*j-self.ecp*2,0)))
                          two_elec += openfermion.FermionOperator(((2*a-self.ecp*2,1),    (2*b+1-self.ecp*2,1),   (2*i-self.ecp*2,0),    (2*j+1-self.ecp*2,0)))
                          two_elec += openfermion.FermionOperator(((2*a-self.ecp*2,1),  (2*b+1-self.ecp*2,1),     (2*i+1-self.ecp*2,0),  (2*j-self.ecp*2,0)))
                          two_elec += openfermion.FermionOperator(((2*a+1-self.ecp*2,1),    (2*b-self.ecp*2,1),   (2*i-self.ecp*2,0),    (2*j+1-self.ecp*2,0)))
                          norm = 0
                          two_elec -= openfermion.hermitian_conjugated(two_elec)


                          two_elec = openfermion.normal_ordered(two_elec)


                          for term in two_elec.terms:

                              norm += two_elec.terms[term]*two_elec.terms[term]
                          if two_elec.many_body_order()>0 and norm!=0:
                              self.SQ_Doubles.append(two_elec/np.sqrt(norm))
                              self.Doubles.append([a-self.ecp,b-self.ecp,i-self.ecp,j-self.ecp])               
                          
                          two_elec = openfermion.FermionOperator(((2*a+1-2*self.ecp,1),(2*b-self.ecp*2,1),(2*i+1-self.ecp*2,0),(2*j-self.ecp*2,0)))
                          two_elec += openfermion.FermionOperator(((2*a-self.ecp*2,1),(2*b+1-self.ecp*2,1),(2*i-self.ecp*2,0),(2*j+1-self.ecp*2,0)))
                          two_elec -= openfermion.FermionOperator(((2*a-self.ecp*2,1),(2*b+1-self.ecp*2,1),(2*i+1-self.ecp*2,0),(2*j-self.ecp*2,0)))
                          two_elec -= openfermion.FermionOperator(((2*a+1-self.ecp*2,1),(2*b-self.ecp*2,1),(2*i-self.ecp*2,0),(2*j+1-self.ecp*2,0)))
                          norm = 0
                          two_elec -= openfermion.hermitian_conjugated(two_elec)
                          two_elec = openfermion.normal_ordered(two_elec)

                          for term in two_elec.terms:
                              norm += two_elec.terms[term]*two_elec.terms[term]

                          if two_elec.many_body_order()>0 and norm!=0:
                              self.SQ_Doubles.append(two_elec/np.sqrt(norm))

                              self.Doubles.append([a-self.ecp,b-self.ecp,i-self.ecp,j-self.ecp])

         if self.intra!=None:
              random.seed(int(self.intra))
              c = list(zip(self.SQ_Doubles, self.Doubles))
              random.shuffle(c)
              self.SQ_Doubles, self.Doubles = zip(*c)


    def SD_Triplet(self):
         #Singles
         occs = [i for i in range(0, math.ceil(self.molecule.n_electrons/2)+math.floor((self.molecule.multiplicity-1)/2))]
         
         for i in range(0, len(occs)):
             for a in range(len(occs), self.molecule.n_orbitals):
                 one_elec = (1/np.sqrt(2))*openfermion.FermionOperator(((2*a-self.ecp*2,1),(2*i-self.ecp*2,0)))
                 one_elec += (1/np.sqrt(2))*openfermion.FermionOperator(((2*a+1-self.ecp*2,1),(2*i+1-self.ecp*2,0)))
                 one_elec -= openfermion.hermitian_conjugated(one_elec)
                 one_elec = openfermion.normal_ordered(one_elec)
                 norm = 0
                 for term in one_elec.terms:

                     norm += one_elec.terms[term]*one_elec.terms[term]
                 if one_elec.many_body_order()>0 and norm!=0:

                     self.SQ_Singles.append(one_elec/np.sqrt(norm))
                     self.Singles.append([a-self.ecp,i-self.ecp])               
         
         if self.intra!=None:
              random.seed(int(self.intra))
              c = list(zip(self.SQ_Singles, self.Singles))
              random.shuffle(c)
              self.SQ_Singles, self.Singles = zip(*c)
 
              
         #Doubles
         pairs = []
         for j in range(int(self.ecp), len(occs)):
             for i in range(j, len(occs)):
                 for b in range(len(occs), self.molecule.n_orbitals):
                     for a in range(b, self.molecule.n_orbitals):
                          two_elec = 2*openfermion.FermionOperator(((2*a-self.ecp*2,1),   (2*b-self.ecp*2,1),     (2*i-self.ecp*2,0),    (2*j-self.ecp*2,0)))
                          two_elec += 2*openfermion.FermionOperator(((2*a+1-self.ecp*2,1),(2*b+1-self.ecp*2,1),   (2*i+1-self.ecp*2,0),  (2*j+1-self.ecp*2,0)))
                          two_elec += openfermion.FermionOperator(((2*a+1-self.ecp*2,1),  (2*b-self.ecp*2,1),     (2*i+1-self.ecp*2,0),  (2*j-self.ecp*2,0)))
                          two_elec += openfermion.FermionOperator(((2*a-self.ecp*2,1),    (2*b+1-self.ecp*2,1),   (2*i-self.ecp*2,0),    (2*j+1-self.ecp*2,0)))
                          two_elec += openfermion.FermionOperator(((2*a-self.ecp*2,1),  (2*b+1-self.ecp*2,1),     (2*i+1-self.ecp*2,0),  (2*j-self.ecp*2,0)))
                          two_elec += openfermion.FermionOperator(((2*a+1-self.ecp*2,1),    (2*b-self.ecp*2,1),   (2*i-self.ecp*2,0),    (2*j+1-self.ecp*2,0)))
                          norm = 0
                          two_elec -= openfermion.hermitian_conjugated(two_elec)
                          two_elec = openfermion.normal_ordered(two_elec)
                          for term in two_elec.terms:
                              norm += two_elec.terms[term]*two_elec.terms[term]
                          if two_elec.many_body_order()>0 and norm!=0:
                              self.SQ_Doubles.append(two_elec/np.sqrt(norm))
                              self.Doubles.append([a-self.ecp,b-self.ecp,i-self.ecp,j-self.ecp])               
                          
                          two_elec = openfermion.FermionOperator(((2*a+1-2*self.ecp,1),(2*b-self.ecp*2,1),(2*i+1-self.ecp*2,0),(2*j-self.ecp*2,0)))
                          two_elec += openfermion.FermionOperator(((2*a-self.ecp*2,1),(2*b+1-self.ecp*2,1),(2*i-self.ecp*2,0),(2*j+1-self.ecp*2,0)))
                          two_elec -= openfermion.FermionOperator(((2*a-self.ecp*2,1),(2*b+1-self.ecp*2,1),(2*i+1-self.ecp*2,0),(2*j-self.ecp*2,0)))
                          two_elec -= openfermion.FermionOperator(((2*a+1-self.ecp*2,1),(2*b-self.ecp*2,1),(2*i-self.ecp*2,0),(2*j+1-self.ecp*2,0)))
                          norm = 0
                          two_elec -= openfermion.hermitian_conjugated(two_elec)
                          two_elec = openfermion.normal_ordered(two_elec)
                          for term in two_elec.terms:
                              norm += two_elec.terms[term]*two_elec.terms[term]

                          if two_elec.many_body_order()>0 and norm!=0:
                              self.SQ_Doubles.append(two_elec/np.sqrt(norm))
                              self.Doubles.append([a-self.ecp,b-self.ecp,i-self.ecp,j-self.ecp])

         if self.intra!=None:
              random.seed(int(self.intra))
              c = list(zip(self.SQ_Doubles, self.Doubles))
              random.shuffle(c)
              self.SQ_Doubles, self.Doubles = zip(*c)

    def Screen_Commutators(self):

        #Singles
        New_Singles = []
        New_SQ_Singles = []
        New_JW_Singles = []
        for single in range(0, len(self.Full_JW_Ops)):
            if self.HF_ket.transpose().conj().dot(self.JW_hamiltonian).dot(self.Full_JW_Ops[single]).dot(self.HF_ket).toarray()[0][0] != 0:
                New_SQ_Singles.append(self.Full_Ops[single])
                New_JW_Singles.append(self.Full_JW_Ops[single])
        self.Full_Ops = New_SQ_Singles
        self.Full_JW_Ops = New_JW_Singles
  
    def Spin_Adapt(self):
        New_SQ_Ops = []
        New_Ops = []
        done = []
        #Singles
        for chi_i in range(0, self.molecule.n_orbitals):
            for chi_a in range(chi_i+1, self.molecule.n_orbitals):
                if ([chi_a*2+1, chi_i*2+1] not in self.Full_Ops):
                    continue

                New_SQ_Ops.append(1/np.sqrt(2)*self.Full_SQ_Ops[(self.Full_Ops.index([chi_a*2+1, chi_i*2+1]))]+1/np.sqrt(2)*self.Full_SQ_Ops[(self.Full_Ops.index([chi_a*2, chi_i*2]))])
                New_Ops.append([chi_a, chi_i])
                done.append(self.Full_Ops.index([chi_a*2, chi_i*2]))
                done.append(self.Full_Ops.index([chi_a*2+1, chi_i*2+1]))
        
        #Doubles
        for op in self.Full_Ops:
            if self.Full_Ops.index(op) not in done:
                if len(op)==2:
                    continue
                a, i, b, j = op
                a = 2*math.floor(a/2)
                i = 2*math.floor(i/2)
                b = 2*math.floor(b/2)
                j = 2*math.floor(j/2)
                if len(set([a,i,b,j]))==4 or (len(set([a,i,b,j]))==3 and (b == i or a == i or b == j)):
                    ind_1 = self.Full_Ops.index([a+1,i+1,b+1,j+1])
                    ind_2 = self.Full_Ops.index([a,i,b,j])
                    ind_3 = self.Full_Ops.index([a+1,i+1,b,j])
                    
                    if [a,i,b+1,j+1] in self.Full_Ops:
                        ind_4 = self.Full_Ops.index([a,i,b+1,j+1])
                        sign_4 = 1
                    elif [b+1, i, a, j+1] in self.Full_Ops:
                        ind_4 = self.Full_Ops.index([b+1,i,a,j+1])
                        sign_4 = -1
                    elif [b+1, j+1, a, i] in self.Full_Ops:
                        ind_4 = self.Full_Ops.index([b+1,j+1,a,i])
                        sign_4 = 1 

                    if [a,i+1,b+1,j] in self.Full_Ops:
                        ind_5 = self.Full_Ops.index([a,i+1,b+1,j])
                        sign_5 = -1
                    elif [b+1,i+1,a,j]:
                        ind_5 = self.Full_Ops.index([b+1,i+1,a,j])
                        sign_5 = 1

                    if [a+1, i, b, j+1] in self.Full_Ops:
                        ind_6 = self.Full_Ops.index([a+1,i,b,j+1])
                        sign_6 = -1
                    elif [a+1, j+1, b, i] in self.Full_Ops:
                        ind_6 = self.Full_Ops.index([a+1,j+1,b,i])
                        sign_6 = 1
                    elif [i, a+1, j+1, b] in self.Full_Ops:
                        ind_6 = self.Full_Ops.index([i,a+1,j+1,b])
                        sign_6 = 1



                    New_SQ_Ops.append(1/np.sqrt(12)*(2*self.Full_SQ_Ops[ind_1]+2*self.Full_SQ_Ops[ind_2]+self.Full_SQ_Ops[ind_3]+sign_4*self.Full_SQ_Ops[ind_4]-sign_5*self.Full_SQ_Ops[ind_5]-sign_6*self.Full_SQ_Ops[ind_6]))
                    New_Ops.append([int(a/2),int(i/2),int(b/2),int(j/2)])
                    New_SQ_Ops.append(.5*(self.Full_SQ_Ops[ind_3]+sign_4*self.Full_SQ_Ops[ind_4]+sign_5*self.Full_SQ_Ops[ind_5]+sign_6*self.Full_SQ_Ops[ind_6])) 
                    New_Ops.append([int(a/2),int(i/2),int(b/2),int(j/2)])
                    done.append(ind_1)
                    done.append(ind_2) 
                    done.append(ind_3)
                    done.append(ind_4)
                    done.append(ind_5)
                    done.append(ind_6)

                elif len(set([a,i,b,j]))==3 or (len(set([a,i,b,j]))==2 and [a,i,b,j].count(a)!=2):
                    if [a, i, b+1, j+1] in self.Full_Ops:
                        ind1 = self.Full_Ops.index([a, i, b+1, j+1])
                        sign_1 = 1
                    elif [i, a, j+1, b+1] in self.Full_Ops:
                        ind1 = self.Full_Ops.index([i,a,j+1,b+1])
                        sign_1 = -1
                    elif [b+1, j+1, a, i] in self.Full_Ops:
                        ind1 = self.Full_Ops.index([b+1, j+1, a, i])
                        sign_1 = 1
                    elif [a, j+1, b+1, i] in self.Full_Ops:
                        ind1 = self.Full_Ops.index([a, j+1, b+1, i])
                        sign_1 = -1
                    elif [b+1, i, a, j+1] in self.Full_Ops:
                        ind1 = self.Full_Ops.index([b+1, i, a, j+1])
                        sign_1 = -1

                                            
                    if [a+1, i+1, b, j] in self.Full_Ops:
                        ind2 = self.Full_Ops.index([a+1,i+1,b,j])
                        sign_2 = 1
                    elif [i+1, a+1, j, b] in self.Full_Ops:
                        ind2 = self.Full_Ops.index([i+1,a+1,j,b])
                        sign_2 = -1
                    New_SQ_Ops.append(1/np.sqrt(2)*(sign_1*self.Full_SQ_Ops[ind1]+sign_2*self.Full_SQ_Ops[ind2]))
                    New_Ops.append([int(i/2), int(a/2), int(j/2), int(b/2)])
                    done.append(ind1)
                    done.append(ind2)
                

                elif len(set([a,i,b,j]))==2:
                    New_SQ_Ops.append(self.Full_SQ_Ops[self.Full_Ops.index(op)])
                    New_Ops.append([int(a/2), int(i/2), int(b/2), int(j/2)])
                    done.append(self.Full_Ops.index(op))

        assert(len((done)) == len(set(done)))
        assert(len((done)) == len((self.Full_SQ_Ops)))
        self.Full_SQ_Ops = []
        self.Full_Ops = []                             
        for op in range(0, len(New_SQ_Ops)):
            op2 = openfermion.normal_ordered(New_SQ_Ops[op])
            if op2.many_body_order()>0:
                self.Full_SQ_Ops.append(op2)
                self.Full_Ops.append(New_Ops[op])

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
        self.milestones = []
        self.indices = []
        self.parameters = []
    def __str__(self):
        return(str(self.Full_Ops))
    def dump(self, filename):
        dst = open(filename, 'w')
        for i in range(0, len(self.indices)):
            dst.write(str(self.indices[i])+' '+str(self.parameters[i])+'\n')
    def read(self, filename, ops):
        src = open(filename, 'r')
        for i in src.readlines():
            num = int(i.split()[0])
            param = float(i.split()[1])
            self.Full_JW_Ops.append(ops.Full_JW_Ops[num])
            self.Full_Ops.append(ops.Full_Ops[num])
            self.indices.append(num)
            self.parameters.append(param)


