from context import *

import numpy as np
import numpy.random 
import time
import pandas
import pickle
import logging

from qiskit import Aer, QuantumCircuit
from qiskit.aqua.operators import ExpectationFactory, CircuitSampler, CircuitStateFn, StateFn
from qiskit.aqua.components.optimizers import SLSQP, SPSA
from qiskit.aqua.algorithms import VQE

import sys

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 



def random_disordered_magnetic_fields(W, num_qubits, random_seed=42):
	numpy.random.seed(random_seed)
	potentials = numpy.random.rand(num_qubits)-0.5
	potentials *= 2*W
	return hamiltonian.magnetic_fields(potentials)

def random_disordered_hamiltonian(W, num_qubits, potentials):
	return hamiltonian.heisenberg1D(num_qubits)+hamiltonian.magnetic_fields(potentials*W)


Wstring = sys.argv[1]
Wrange = [int(Wpart) for Wpart in Wstring.split(',')]
num_qubits = 6
num_trials = 200
reps = 1
maxiter = 1000
entanglement = 'sca'
random_seed_base = 142857
optimizer_name = 'VVQE2_SLSQP'
#optimizer = SPSA(max_trials = maxiter)
optimizer = SLSQP(maxiter=maxiter)
backend_name = 'statevector_simulator'
evaluation_type = 'statevector'
backend_method = 'statevector'



for Wval in Wrange:
	data = []
	name = "W%s_q%s_VVQE_SLSQP_%s_rep%s"%(Wval,num_qubits,entanglement,reps)
	print("Saving to file {}".format(name+".pkl"))
	logging.basicConfig(filename='run_multiple_tests_at_W{}_q{}_reps{}.log'.format(Wval,num_qubits,reps), level=logging.INFO)

	with open("q{}_1000potentials.pkl".format(num_qubits),'rb') as f:
		potentials = pickle.load(f)
	for potential in potentials:
		assert len(potential) == num_qubits
	for i in range(num_trials):
		#random_seed = Wval*89 + i*61 + random_seed_base
		#H = random_disordered_hamiltonian(Wval, num_qubits, random_seed)
		potential = np.array(potentials[i])
		H = random_disordered_hamiltonian(Wval, num_qubits, potential)
		ans = variational_form.sz_conserved_ansatz(num_qubits, reps=reps, entanglement=entanglement, spindn_cluster = 'balanced')
		
		
		energy, variance, fidelity, time_elapsed, vqe_results = combined_optimization.combined_optimizer(H, ans, optimizer, backend_name = backend_name, evaluation_type = evaluation_type)
		
		data_dict={'W':Wval,'elapsed_time':time_elapsed,
						'reps':reps,'entanglement':entanglement,
						'opt_params':vqe_results['optimal_point'],
						'Opt':optimizer_name,
						'statevector':vqe_results['eigenstate'],
						'E':energy,'Var':variance,'fidelity':fidelity}
		data.append(data_dict)
		print("COMPLETED RUN #{}/{} AT W={} ({}s)".format(i+1,num_trials, Wval,time_elapsed))
	with open(name+".pkl",'wb') as f:
		pickle.dump(data,f)
	assert len(data) == num_trials

	


		






