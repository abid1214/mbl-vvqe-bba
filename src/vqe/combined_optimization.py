from vvqe import *
import hamiltonian as ham
import variational_form as vf
from qiskit import Aer
from qiskit.aqua.operators.expectations import ExpectationFactory
from qiskit.aqua.operators.state_fns import StateFn, CircuitStateFn
from qiskit.aqua.operators import CircuitSampler
from qiskit.aqua.algorithms import VQE
import time
import numpy.random

def random_initial_point(num_points):
	#TODO: Change this to a better random function?
	return (numpy.random.rand(num_points)-0.5)*4*np.pi

def expectation(wavefn, operator, backend):
	#Computes the expectation value on an operator with respect to an ansatz that has its parameters filled in
    exp = ExpectationFactory.build(operator = operator, backend = backend)
    composed_circuit = exp.convert(StateFn(operator, is_measurement=True)).compose(CircuitStateFn(wavefn))
    sampler = CircuitSampler(backend)
    vals = sampler.convert(composed_circuit).eval()
    return np.real(vals)

def attach_parameters(ansatz, params):
	#Assigns a list of parameters to an ansatz
	assert len(params) == ansats.num_parameters
	param_dict = dict(zip(ansatz.parameters, params))
	return ansatz.assign_parameters(param_dict)

def combined_optimizer(hamiltonian, ansatz, optimizer, backend_name = 'statevector_simulator', evaluation_type = 'none'):
	#First optimizes H^2 using VQE, then optimizes the variance using VVQE
	hamiltonian_squared = ham.square(hamiltonian)

	backend = Aer.get_backend(backend_name)
	t0 = time.process_time()
	h2_algorithm = VQE(hamiltonian_squared, ansatz, optimizer, include_custom=True, initial_point = random_initial_point(ansatz.num_parameters))
	h2_results = h2_algorithm.run(backend)
	h2_optimal_params = h2_results['optimal_point']

	variance_algorithm = VVQE(hamiltonian, ansatz, optimizer, include_custom=True, initial_point = h2_optimal_params)
	variance_results = variance_algorithm.run(backend)
	t1 = time.process_time()

	if evaluation_type == 'statevector':
		#Turns the optimal wavefunction into a statevector to compute energy, variance, fidelity, etc.
		assert backend_name == 'statevector_simulator'
		vec = variance_results['eigenstate']

		H_mat = hamiltonian.to_matrix()
		# H^2
		H_squared_mat = hamiltonian_squared.to_matrix()
		(evals, evecs)   = np.linalg.eigh(H_mat)
		energy_linear   = (vec.conj()@H_mat@vec).real
		variance_linear = (vec.conj()@H_squared_mat@vec-energy_linear**2).real
		fidelity=np.max(np.abs(evecs.T@vec))
		return energy_linear, variance_linear, fidelity, t1-t0, variance_results
	else if evaluation_type == 'circuit'
		#Computes energy and variance using circuit operators. Can't compute fidelity
		final_wavefunction = attach_parameters(ansatz, variance_results['optimal_point'])
		energy = expectation(final_wavefunction, hamiltonian, backend)[0]
		variance = expectation(final_wavefunction, hamiltonian_squared, backend)[0] - energy**2
		return energy, variance, -1, t1-t0, variance_results
	#If no evaluation type, just returns the VQE Results object
	return variance_results
	
  	




	