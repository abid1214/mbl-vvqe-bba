from qiskit.circuit.library import ExcitationPreserving
from qiskit import QuantumCircuit
import math


def sz_conserved_ansatz(num_qubits, 
	insert_barriers = False,
	entanglement = 'sca',
	reps = 1,
	total_spinup = -1, 
	spinup_cluster = 'balanced'):
	ansatz = ExcitationPreserving(num_qubits, insert_barriers = insert_barriers, entanglement = entanglement, reps = reps)
	ansatz_circuit = QuantumCircuit(num_qubits)

	if total_spinup = -1:
		total_spinup = math.ceil(num_qubits/2.)
	if total_spinup > num_qubits:
		print("WARNING: total Sz {0} exceeds qubit number {1}".format(total_spinup, num_qubits))
		total_spinup = num_qubits
	if spinup_cluster = 'left_clustered':
		for i in range(total_spinup):
			ansatz_circuit.x(i)
	if spinup_cluster = 'balanced':
		spinup_interval = num_qubits/total_spinup
		for i in range(total_spinup):
			ansatz_circuit.x(math.floor(i*spinup_interval))
	ansatz_circuit.compose(ansatz, inplace = True)
	return ansatz_circuit

