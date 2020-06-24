import numpy as np

from qiskit.quantum_info.operators.pauli import Pauli
from qiskit.aqua.operators.primitive_ops.pauli_op import PauliOp

def heisenberg1D(num_qubits):
    # Returns a qiskit Operator representing
    # the 1D Heisenberg model with open boundary conditions:
    #      1/4 * \sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
    
    ind_op = 0
    coeff  = 0.25
    for i in range(num_qubits-1):
        for pauli_label in ['X', 'Y', 'Z']:
            labels   = ['I']*(i) + [pauli_label, pauli_label] + ['I']*(num_qubits-(i+2))

            pauli    = Pauli(label=labels)
            pauli_op = PauliOp(pauli, coeff)

            if ind_op == 0:
                hamiltonian = pauli_op
            else:
                hamiltonian += pauli_op

            ind_op += 1
                
    return hamiltonian

def magnetic_fields(potentials):
    # Returns a qiskit Operator representing
    # the magnetic fields
    #      1/2 * \sum_i h_i Z_i
    # where h_i is the vector specified by potentials.
    
    num_qubits = len(potentials)
    
    for i in range(num_qubits):
        coeff  = 0.5 * potentials[i]
        labels = ['I']*(i) + ['Z'] + ['I']*(num_qubits-(i+1))

        pauli    = Pauli(label=labels)
        pauli_op = PauliOp(pauli, coeff)

        if i == 0:
            op = pauli_op
        else:
            op += pauli_op

    return op

def square(operator):
    # Returns a qiskit ComposedOp Operator
    # representing the operator squared.
    return operator.compose(operator)
