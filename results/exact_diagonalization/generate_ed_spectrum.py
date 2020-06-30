import numpy as np
import hamiltonian,variational_form,vvqe
from hamiltonian import *
from variational_form import sz_conserved_ansatz
from qiskit import *
from qiskit.aqua.components.optimizers import *
from qiskit.aqua.algorithms import VQE
from qiskit.quantum_info import entropy,partial_trace
from vvqe import VVQE
import time
import datetime
import pickle
import sys
import gzip

def ptrace(L,LA,psi):
    """compute partial trace of density matrix
    Args:
        L: total system size
        LA: subsystem of reduced density matrix
        psi: wavefunction

    Returns:
        ptr_rho: reduced density matrix on right-hand subsystem size LA
      (qiskit notation)
    """
    rho = np.outer(np.conjugate(psi),psi) 
    sizeA = 2**LA
    sizeB = 2**(L-LA)
    ptr_rho = np.zeros((sizeA,sizeA),dtype=complex)
    for k in range(sizeA):
        for l in range(sizeA):
            for j in range(sizeB):
                ptr_rho[k,l]=ptr_rho[k,l]+rho[k*sizeB+j,l*sizeB+j] 
    
    return ptr_rho


def entropy(L,LA,psi):
  """compute entanglement entropy of reduced density matrix
  Args:
    L: total system size
    LA: subsystem of reduced density matrix
    psi: wavefunction
  
  Returns:
    EE: 2nd renyi entropy reduced density matrix on right-hand subsystem size LA
    vEE: von neumann entanglement entropy reduced density matrix on right-hand 
      subsystem size LA
  """
  ptr_rho = ptrace(L,LA,psi)
  _,s,_ = np.linalg.svd(ptr_rho,full_matrices=False)
  s = s[s>1e-12]
  vEE = -np.sum(np.multiply(s, np.log2(s))) # Von Neumann entropy
  s = np.square(np.abs(s))
  EE = -np.log(np.sum(s)) # 2nd renyi entropy
  return EE, vEE

# to generate data files directly, use this

#np.random.seed(42)
#for num_qubits in range(4,12,2):
#    num_trials      = 1000           # total # of trials
#    start           = 0             # first trial #
#    
#    # open saved potential files
#    # using the pickle instead of text in order to prevent precision problems
#    with open("results/q{}_1000potentials.pkl".format(num_qubits),'rb') as f:
#        allPotentials = pickle.load(f)
#    for potential in allPotentials:
#        assert len(potential) == num_qubits
#        
#    allData = []
#    # cache hamiltonian
#    HJ      = heisenberg1D(num_qubits)
#    H_Jterm = HJ.to_matrix()
#    
#    for W in [1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8]:
#        
#        print("Start q={} W={} {}".format(num_qubits,W,datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
#        dictList = [{}]*num_trials
#        
#        for trial in range(start,start+num_trials):
#            potentials = allPotentials[trial]
#            # Hamiltonian H
#            Hw             = W * magnetic_fields(potentials)
#            H_mat          = H_Jterm+Hw.to_matrix()
#            ## Eigenvalue of H
#            try:
#                evals, evecs   = np.linalg.eigh(H_mat)
#            except:
#                continue
#            dictList[trial-start] = ({"W":W,"evals":evals,"evecs":evecs,"realization":trial})
#        allData.append(dictList)
#    
#    pickle.dump(allData,gzip.open("results/ed_spectrum_data_{}_qubits.pkl.gz".format(num_qubits),'wb'))

#to generate the pandas file use this
np.random.seed(42)
entropyData = []
for num_qubits in range(4,10,2):
    num_trials      = 1000           # total # of trials
    start           = 0             # first trial #
    
    # open saved potential files
    # using the pickle instead of text in order to prevent precision problems
    with open("results/q{}_1000potentials.pkl".format(num_qubits),'rb') as f:
        allPotentials = pickle.load(f)
    for potential in allPotentials:
        assert len(potential) == num_qubits
        
    allData = []
    # cache hamiltonian
    HJ      = heisenberg1D(num_qubits)
    H_Jterm = HJ.to_matrix()
    
    for W in [1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8]:
        
        print("Start q={} W={} {}".format(num_qubits,W,datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        
        
        for trial in range(start,start+num_trials):
            potentials = allPotentials[trial]
            # Hamiltonian H
            Hw             = W * magnetic_fields(potentials)
            H_mat          = H_Jterm+Hw.to_matrix()
            ## Eigenvalue of H
            try:
                evals, evecs   = np.linalg.eigh(H_mat)
            except:
                continue
            for iE,(E,vec) in enumerate(zip(evals,evecs.T)):
                for LA in range(1,num_qubits//2+1):
                    S_vn,S_2 = entropy(num_qubits,LA,vec)
                    entropyData.append({"W":W,"Energy":E,"realization":trial,
                                    "S1":S_vn,"S2":S_2,"LA":LA,"E_order":iE})

import pandas as pd
df = pd.DataFrame(entropyData)
df.to_pickle("ed_entropyData_df.pkl")
