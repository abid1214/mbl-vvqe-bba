import numpy as np
import time

from .context import variational_form as vf, hamiltonian as ham

from qiskit import Aer, QuantumCircuit
from qiskit.aqua.components.optimizers import *
from qiskit.aqua.operators import ExpectationFactory, CircuitSampler, CircuitStateFn, StateFn

from ..vvqe import VVQE

##### TEST #####
def test_vvqe():
  np.random.seed(42)

  num_qubits = 2
  potentials = 2.0*np.random.rand(num_qubits) - 1.0
  W          = 5.0

  # Hamiltonian H
  H = ham.heisenberg1D(num_qubits) + W * ham.magnetic_fields(potentials)

  H2 = ham.square(H)

  reps         = 1
  entanglement = 'full'
  var_form     = vf.sz_conserved_ansatz(num_qubits, reps=reps, entanglement=entanglement, spindn_cluster = 'balanced')

  maxiter1   = 500
  optimizer1 = SLSQP(maxiter=maxiter1, disp=True) #SPSA(max_trials=maxiter1) #COBYLA(maxiter=maxiter1, disp=True)

  backend = Aer.get_backend('statevector_simulator')

  qalgorithm = VVQE(H, var_form, optimizer1, include_custom=True)

  t0          = time.process_time()
  opt_result1 = qalgorithm.run(backend)
  t1          = time.process_time()
  print('TIME ELAPSED: {}'.format(t1-t0))

  H_mat = H.to_matrix()
  # H^2
  H_squared_mat = H2.to_matrix()
  ## Eigenvalue of H closest to zero is 0.01933818
  (evals, evecs)   = np.linalg.eigh(H_mat)
  ## Eigenvalue of H^2 closest to zero is 3.73965165e-04 == (0.01933818)**2.0
  (evals2, evecs2) = np.linalg.eigh(H_squared_mat)
  assert(np.allclose(np.sort(np.abs(evals)**2.0), evals2))

  vec = opt_result1['eigenstate']
  energy   = (vec.conj()@H_mat@vec).real
  variance = (vec.conj()@H_squared_mat@vec-energy**2).real
  fidelity=np.max(np.abs(evecs.T@vec))
  print("E={} Var={} Fidelity={}".format(energy,variance,fidelity))
  #test variance matches
  assert abs(variance-opt_result1['optimal_value']) < 1e-8
  # test sensibility
  assert 0.5 > variance > 0
  assert fidelity<1 and fidelity > 0.9
