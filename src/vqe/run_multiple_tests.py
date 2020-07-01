import numpy as np
import numpy.random 
import time
import pandas as pd
import pickle
import logging

from qiskit import Aer, QuantumCircuit
from qiskit.aqua.operators import ExpectationFactory, CircuitSampler, CircuitStateFn, StateFn
from qiskit.aqua.components.optimizers import SLSQP, SPSA, COBYLA
from qiskit.aqua.algorithms import VQE

import sys
import os

import hamiltonian
import variational_form

import combined_optimization

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def collect_pickles(base_name, out_name):
	"""
		Takes .pkl files of individual runs and collects them into a single .pkl file
		Does not remove the original .pkl files
	"""
	max_numbers_to_check = 5000
	if '~' not in base_name:
		print("Error: Could not find wildcard in base name")
	else:
		pickles = []
		for i in range(max_numbers_to_check):
			pickle_file_name = base_name.replace('~', str(i))
			if os.path.isfile(pickle_file_name):
				pickles += pd.read_pickle(pickle_file_name)
				#pickles.append(pd.DataFrame(pd.read_pickle(pickle_file_name)))
		#combined_frame = pd.concat(pickles)
		with open(out_name, "wb") as f:
			#pickle.dump(combined_frame, f)
			pickle.dump(pickles, f)


def random_disordered_hamiltonian(W, num_qubits, potentials):
	"""Creates the Heisenberg Hamiltonian Operator with size num_qubits and disordered field potentials*W"""
	return hamiltonian.heisenberg1D(num_qubits)+hamiltonian.magnetic_fields(potentials*W)



"""
	Computes runs of certain W values and number of qubits and stores the results/data in a .pkl file
	Input: 
		String of the form W1,W2,W3...,WN
		Specifies the different W values
	Output:
		Creates a .pkl file at each W value
		Keeps individual .pkl files for each specific trial

"""
Wstring = sys.argv[1] 
Wrange = [float(Wpart) for Wpart in Wstring.split(',')]
num_qubits = 6
num_trials = 25
num_trials_start = 0
reps = 1
maxiter = 1000
entanglement = 'full'
random_seed_base = 142857
# Different optimizer types/names to enable/disable as needed
#optimizer = SPSA(max_trials = maxiter)
#optimizer_name = 'VVQE2_SPSA'

#optimizer = COBYLA(maxiter = maxiter, disp=True)
#optimizer_name = 'VVQE2_COBYLA'
optimizer = SLSQP(maxiter=maxiter, disp=True)
optimizer_name = 'SLSQP'
backend_name = 'statevector_simulator'
evaluation_type = 'statevector'
#backend_method = 'automatic'
backend_options = {'method': 'automatic'}



for Wval in Wrange:

	if Wval.is_integer():
		Wval = int(Wval)
	name = "../../results/vvqe/W%s_q%s_VVQE_%s_%s_rep%s"%(Wval,num_qubits,optimizer_name,entanglement,reps)
	logging.basicConfig(filename='run_multiple_tests_at_W{}_q{}_reps{}.log'.format(Wval,num_qubits,reps), level=logging.INFO)

	with open("../../results/q{}_1000potentials.pkl".format(num_qubits),'rb') as f:
		potentials = pickle.load(f)
	for potential in potentials:
		assert len(potential) == num_qubits
	for i in range(num_trials_start, num_trials):
		base_name = "../../results/vvqe/W%s_q%s_VVQE_%s_%s_rep%s"%(Wval,num_qubits,optimizer_name,entanglement,reps)
		name = base_name + str(i)
		if os.path.isfile(name+".pkl"):
			print("File {} already found".format(name+".pkl"))
			continue
		data = []
		potential = np.array(potentials[i])
		H = random_disordered_hamiltonian(Wval, num_qubits, potential)
		ans = variational_form.sz_conserved_ansatz(num_qubits, reps=reps, entanglement=entanglement, spindn_cluster = 'balanced')
		
		energy, variance, fidelity, time_elapsed, vqe_results = combined_optimization.combined_optimizer(H, ans, optimizer, 
			backend_name = backend_name, backend_options=backend_options, 
			include_custom = True, num_shots = 1, evaluation_type = evaluation_type)
		
		data_dict={'W':Wval,'elapsed_time':time_elapsed,
						'reps':reps,'entanglement':entanglement,
						'opt_params':vqe_results['optimal_point'],
						'Opt':"VVQE2_"+optimizer_name,
						'statevector':vqe_results['eigenstate'],
						'E':energy,'Var':variance,'fidelity':fidelity}
		data.append(data_dict)
		print("COMPLETED RUN #{}/{} AT W={} ({}s, Var={})".format(i+1,num_trials, Wval,time_elapsed, variance))
		with open(name+".pkl",'wb') as f:
			pickle.dump(data,f)
	collect_pickles(base_name+"~.pkl",base_name + ".pkl")


	


		






