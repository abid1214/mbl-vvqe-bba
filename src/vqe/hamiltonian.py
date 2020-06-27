import numpy as np

from qiskit.quantum_info.operators.pauli import Pauli
from qiskit.aqua.operators.primitive_ops.pauli_op import PauliOp

def identity(num_qubits):
    # Returns an identity operator.
    labels   = ['I']*num_qubits
    pauli    = Pauli(label=labels)
    pauli_op = PauliOp(pauli, 1.0)

    return pauli_op
    
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

def sum_duplicates(operator):
    # Sums together the coefficients
    # of duplicated PauliOps.
    # Note: Assumes a SummedOp of PauliOp.

    if len(operator) == 0:
        return operator
    
    op_terms = dict()
    for pauliop in operator:
        pauli = pauliop.primitive
        if pauli in op_terms:
            op_terms[pauli] += pauliop.coeff
        else:
            op_terms[pauli] = pauliop.coeff

    first_term = True
    for key in op_terms:
        pauliop = PauliOp(key, op_terms[key])
        if first_term:
            op         = pauliop
            first_term = False
        else:
            op += pauliop

    return op

def remove_zeros(operator, tol=1e-15):
    # Removes PauliOps with coefficients
    # of zero.
    # Notes: Assumes a SummedOp of PauliOp. 
    # Cannot handle a zero operator.
    
    first_term = True
    for pauliop in operator:
        if np.abs(pauliop.coeff) > tol:
            if first_term:
                op         = pauliop
                first_term = False
            else:
                op  += pauliop

    if first_term:
        raise ValueError('Operator is the zero operator!')
    
    return op

def square(operator):
    # Returns a qiskit ComposedOp Operator
    # representing the operator squared.
    op2 = operator.compose(operator)
    op2 = op2.reduce().reduce()

    op2 = sum_duplicates(op2)
    op2 = remove_zeros(op2)

    return op2