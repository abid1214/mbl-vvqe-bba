import numpy as np

from qiskit.quantum_info.operators.pauli import Pauli
from qiskit.aqua.operators.primitive_ops.pauli_op import PauliOp
    
def heisenberg1D(num_qubits):
    '''construct the 1D Heisenberg model with open
       boundary conditions:
         1/4 * \sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})

       Inputs:
         num_qubits the number of qubits (spins) in the chain
    
       Returns:
         Operator (SummedOp of PauliOps) representing the Hamiltonian
    '''
    
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
    '''construct magnetic fields of the form
         1/2 * \sum_i h_i Z_i

       Inputs:
         potentials a list or ndarray representing the potential h_i
    
       Returns:
         Operator (SummedOp of PauliOps) representing the magnetic fields
    '''
    
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
    '''sum together the coefficient of PauliOps
       in a SummedOp of PauliOps with repeated PauliOps.

       Inputs:
         operator a SummedOp of PauliOps
    
       Returns:
         Operator (SummedOp of PauliOps) a new operator with duplicated terms summed together
    '''

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
    '''remove PauliOps from a SummedOp of PauliOps that
       has a zero coefficient.

       Inputs:
         operator a SummedOp of PauliOps
         tol (optional, default=1e-15) the tolerance for what is considered zero
    
       Returns:
         Operator (SummedOp of PauliOps) a new operator with zero terms removed
    '''
    
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
    '''square a SummedOp of PauliOps, making sure to
       sum together identical terms and remove PauliOps
       with zero coefficients.

       Inputs:
         operator a SummedOp of PauliOps
    
       Returns:
         Operator (SummedOp of PauliOps) representing operator * operator
    '''
    
    # Returns a ComposedOp Operator
    # representing the operator squared.
    op2 = operator.compose(operator)
    # Reduce the ComposedOp into a SummedOp of PauliOps.
    op2 = op2.reduce().reduce()

    # Sum the duplicates.
    op2 = sum_duplicates(op2)
    # Remove the zeros.
    op2 = remove_zeros(op2)

    return op2
