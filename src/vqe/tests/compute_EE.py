import numpy as np
import pickle


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


### Test
L = 8
LA = 4
with open('ed_entropy_data_'+str(L)+'_qubits.pkl', 'rb') as f1:
  data1 = pickle.load(f1)

for i in range(1000):
  psi = data1[i]['zero_energy_vec']
  rho = ptrace(L,LA,psi)
  rho_EE, rho_vEE = entropy(L,LA,psi)
  rho_qiskit = data1[i]['reduced_dm'].data
  qiskit_EE = data1[i]['renyi_entropy']
  qiskit_vEE = data1[i]['vn_entropy']
  diff = np.array([i, np.linalg.norm(rho-rho_qiskit)<1e-12,rho_EE-qiskit_EE<1e-12,rho_vEE-qiskit_vEE<1e-12])
  print(diff)
