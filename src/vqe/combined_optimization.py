from vvqe import *
import hamiltonian as ham
import variational_form as vf
from qiskit import Aer, execute
from qiskit.aqua.operators.expectations import ExpectationFactory
from qiskit.aqua.operators.state_fns import StateFn, CircuitStateFn
from qiskit.aqua.operators import CircuitSampler
from qiskit.aqua.algorithms import VQE
import time
import numpy.random
import logging

logger = logging.getLogger(__name__)

def random_initial_point(num_points):
	"""Creates a size num_points numpy array of random values in the range (-2pi, 2pi)"""
	return (numpy.random.rand(num_points)-0.5)*4*np.pi

def expectation(wavefn, operator, backend):
	"""Returns the expectation value of an operator with the provided wavefunction using the provided backend"""
	exp = ExpectationFactory.build(operator = operator, backend = backend)
	composed_circuit = exp.convert(StateFn(operator, is_measurement=True)).compose(CircuitStateFn(wavefn))
	sampler = CircuitSampler(backend)
	vals = sampler.convert(composed_circuit).eval()
	return np.real(vals)

def attach_parameters(ansatz, params):
	"""Given a list of parameters and a parameterized circuit, assigns the parameters to the circuit"""
	assert len(params) == ansatz.num_parameters
	param_dict = dict(zip(ansatz.parameters, params))
	return ansatz.assign_parameters(param_dict)

def get_vector_from_circuit(ansatz, params, backend, label='snap'):
	"""Attempts to use a snapshot to get a vector from an ansatz with the provided list of parameters"""
	snap = ansatz.snapshot(label, snapshot_type='statevector')
	param_dict = dict(zip(ansatz.parameters, params))
	snap_results = execute(ansatz.assign_parameters(param_dict), backend = backend).result()
	vec = snap_results.results[0].data.snapshots.statevector[label][0]
	assert len(vec) == 2**ansatz.num_qubits
	return np.array(vec)

def combined_optimizer(hamiltonian, ansatz, optimizer, 
		num_shots=1, 
		hamiltonian_squared = None, 
		include_custom = True,
		backend_name = 'statevector_simulator', 
		evaluation_type = 'none',
		backend_options = {}):
	"""
	Combines the VQE and VVQE optimization processes to find an eigenstate with energy close to 0 
	with respect to a certain Hamiltonian H. 
	First uses VQE on H^2 to find an energy close to 0, then uses VVQE to refine the variance 
	<H^2> - <H>^2 and get as close to a real eigenstate of the Hamiltonian as possible.

	Args:
		hamiltonian: The Hamiltonian Operator H to optimize
		ansatz: The parameterized QuantumCircuit to optimize on
		optimizer: The classical Optimizer to use
		hamiltonian_squared: If provided, will use that as H^2 instead of creating it from H
		evaluation_type: The way in which returned parameters like energy or variance will be calculated (Default: 'none'):
			'statevector': Finds the statevector of the optimal circuit and diagonalizes the matrix form 
			of the Hamiltonian to obtain the energy, variance and fidelity.
			'circuit': Attempts to find the energy of the optimal circuit by finding its expectation value with H
			without turning it to a statevector first. Fidelity is not calculated and set to -1. Variance is taken from
			the VVQE results.
			'none': Does not calculate energy or fidelity and sets them to -1. Variance is taken from the VVQE results.
		include_custom: Option for the VQE and VVQE calls (Default: True)
		backend_name: The name of the Aer backend to use in VQE/VVQE (Default: 'statevector_simulator')
		backend_options: Extra options to feed into the QuantumInstance passed to VQE/VVQE (Default: empty dict)
	
	Returns:
		Energy <H> of the optimal circuit (-1 if not calculated because evaluation_type='none')
		Variance <H^2> - <H>^2 of the optimal circuit
		Maximum fidelity between the optimal circuit and the eigenstates of the Hamitlonian (-1 if evaluation_type='none' or 'circuit')
		Elapsed time of the optimization process
		The results dictionary returned by the VVQE process
	"""
	
	t0 = time.process_time()
	if hamiltonian_squared == None:
		hamiltonian_squared = ham.square(hamiltonian)

	backend = Aer.get_backend(backend_name)
	quantum_instance = QuantumInstance(Aer.get_backend(backend_name), shots=num_shots, backend_options=backend_options)

	h2_algorithm = VQE(hamiltonian_squared, ansatz, optimizer, include_custom=include_custom, initial_point = random_initial_point(ansatz.num_parameters), quantum_instance = quantum_instance)
	h2_results = h2_algorithm.run()
	h2_optimal_params = h2_results['optimal_point']

	variance_algorithm = VVQE(hamiltonian, ansatz, optimizer, include_custom=include_custom, initial_point = h2_optimal_params, quantum_instance = quantum_instance)
	variance_results = variance_algorithm.run()
	t1 = time.process_time()
	elapsed_time = t1-t0
	fn_evals = variance_results['cost_function_evals'] + h2_results['cost_function_evals']
	
	logger.info("Found variance {} after {} H^2 function calls and {} Variance function calls".format(np.real(variance_results['eigenvalue']), h2_results['cost_function_evals'], variance_results['cost_function_evals']))

	if evaluation_type == 'statevector':
		#Turns the optimal wavefunction into a statevector to compute energy, variance, fidelity, etc.
		vec = np.zeros(2**ansatz.num_qubits)
		if backend_name == 'statevector_simulator':
			vec = variance_results['eigenstate']
		else:
			vec = get_vector_from_circuit(ansatz, variance_results['optimal_point'], backend)
		h_mat = hamiltonian.to_matrix()
		h_squared_mat = hamiltonian_squared.to_matrix()
		(evals, evecs)   = np.linalg.eigh(h_mat)
		energy_linear   = (vec.conj()@h_mat@vec).real
		variance_linear = (vec.conj()@h_squared_mat@vec-energy_linear**2).real
		fidelity=np.max(np.abs(evecs.T@vec))
		elapsed_time = time.process_time()-t0
		logger.info("Energy:{},Var:{},Fid:{},Time:{}".format(energy_linear, variance_linear, fidelity, elapsed_time))
		return energy_linear, variance_linear, fidelity, elapsed_time, variance_results
	elif evaluation_type == 'circuit':
		#Computes energy and variance using circuit operators. Can't compute fidelity
		final_wavefunction = attach_parameters(ansatz, variance_results['optimal_point'])
		energy = expectation(final_wavefunction, hamiltonian, backend)
		variance = variance_results['eigenvalue']
		elapsed_time = time.process_time()-t0
		logger.info("Energy:{},Var:{},Time:{}".format(energy, variance, elapsed_time))
		return energy, variance, -1, elapsed_time, variance_results
	#If no evaluation type, just returns the VQE Results object, elapsed time and variance
	return -1, variance_results['eigenvalue'], -1, elapsed_time, variance_results
	
  	




	