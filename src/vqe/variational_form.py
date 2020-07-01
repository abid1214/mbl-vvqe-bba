from qiskit.circuit.library import ExcitationPreserving
from qiskit import QuantumCircuit
import math
import random


def sz_conserved_ansatz(num_qubits, 
	insert_barriers = False,
	entanglement = 'sca',
	reps = 1,
	total_spindn = -1, 
	spindn_cluster = 'balanced',
	seed = '999999'):
	"""
	Creates a parameterized Quantum Circuit to feed into the VQE and VVQE optimization algorithms.
	Starts with placing spin-flip (X) gates in an unparameterized manner, then applies an ExcitationPreserving circuit to
	parameterize the system without changing the total number of spin ups/downs.

	Args:
		num_qubits: The number of qubits in the circuit
		insert_barriers: Whether a barrier should be placed between the X gates and ExcitationPreserving gates (Default: False)
		total_spindn: How many X gates should be applied initially - if set to -1, will apply floor(num_qubits/2) gates (Default: -1)
		spindn_cluster: Where the X gates should be applied (Default: 'balanced'):
			'balanced': Attempts to distribute the gates evenly along the circuit
			'left_clustered': Places the gates on the first (total_spindn) qubits
			'random': Distributes the gates randomly without applying two gates to the same qubit
		seed: The random seed to use if spindn_cluster = 'random' (Default: 999999)
		entanglement: The type of entanglement that the ExcitationPreserving ansatz should use (Default: 'sca')
		reps: The number of layers of repeating gates that the ExcitationPreserving ansatz should use (Default: 1)

	Returns:
		A parameterized QuantumCircuit 
	"""
	
	
	ansatz = ExcitationPreserving(num_qubits, insert_barriers = insert_barriers, entanglement = entanglement, reps = reps)
	ansatz_circuit = QuantumCircuit(num_qubits)

	if total_spindn == -1:
		total_spindn = math.floor(num_qubits/2.)
	if total_spindn > num_qubits:
		print("WARNING: total Sz {0} exceeds qubit number {1}".format(total_spindn, num_qubits))
		total_spindn = num_qubits

	
	if spindn_cluster == 'left_clustered':
		#Flips all the spins on one side of the circuit
		ansatz_circuit.x(range(total_spindn))
	
	if spindn_cluster == 'balanced':
		#Attempts to distribute the spin flips as evenly as possible
		spindn_interval = num_qubits/total_spindn
		ansatz_circuit.x([math.floor(i*spindn_interval) for i in range(total_spindn)])

	if spindn_cluster == 'random':
		#Flips the spin at random positions (no double flips)
		random.seed(seed)
		spindn_choices = random.sample(range(num_qubits), total_spindn)
		ansatz_circuit.x(spindn_choices)

	if insert_barriers:
		#Adds a barrier between the spin setup gates and the parameterized gates
		ansatz_circuit.barrier()

	ansatz_circuit.compose(ansatz, inplace = True)
	return ansatz_circuit

