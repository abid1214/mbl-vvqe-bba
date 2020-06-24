from qiskit.circuit.library import ExcitationPreserving
from qiskit import QuantumCircuit
import math
import random


def sz_conserved_ansatz(num_qubits, 
	insert_barriers = False,
	entanglement = 'sca',
	reps = 1,
	total_spindn = -1, 
	spindn_cluster = 'balanced'):
	#Returns a QuantumCircuit consisting of spin flips equal to total_spindn, followed by Sz preserving
	#parameter-dependent rotations.
	#If total_spindn = -1, it flips half or floor(half) the spins
	
	ansatz = ExcitationPreserving(num_qubits, insert_barriers = insert_barriers, entanglement = entanglement, reps = reps)
	ansatz_circuit = QuantumCircuit(num_qubits)

	if total_spindn == -1:
		total_spindn = math.floor(num_qubits/2.)
	if total_spindn > num_qubits:
		print("WARNING: total Sz {0} exceeds qubit number {1}".format(total_spindn, num_qubits))
		total_spindn = num_qubits

	
	if spindn_cluster == 'left_clustered':
		#Flips all the spins on one side of the circuit
		for i in range(total_spindn):
			ansatz_circuit.x(i)
	
	if spindn_cluster == 'balanced':
		#Attempts to distribute the spin flips as evenly as possible
		spindn_interval = num_qubits/total_spindn
		for i in range(total_spindn):
			ansatz_circuit.x(math.floor(i*spindn_interval))

	if spindn_cluster == 'random':
		#Flips the spin at random positions (no double flips)
		spindn_choices = random.sample(range(num_qubits), total_spindn)
		for c in spindn_choices:
			ansatz_circuit.x(c)
	
	ansatz_circuit.compose(ansatz, inplace = True)
	return ansatz_circuit

