from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from utilities import *

def _swap_test_init(psi, phi):
    '''helper function for swap_test_QC
       initializes a quantum circuit for swap test
    '''
    num_qubits = psi.num_qubits
    qc = QuantumCircuit(QuantumRegister(1, 'a'),
                        QuantumRegister(num_qubits, 'psi'),
                        QuantumRegister(num_qubits, 'phi'),
                        ClassicalRegister(1, 'm'))
    qc.compose(psi, list(range(1, num_qubits+1)), inplace=True)
    qc.compose(phi, list(range(num_qubits+1,2*num_qubits+1)), inplace=True)
    return qc


def swap_test_QC(psi, phi, idx_list=None):
    ''' swap test algorithm from
        https://en.wikipedia.org/wiki/Swap_test
        Inputs: psi and phi are QuantumCircuit objects
                with the same number of qubits

                if we want a partial overlap of psi and phi,
                idx_list and are the indices
                of the qubits to perform the overlap with
        Return:
            QuantumCircuit object for swap test
    '''
    #initialize registers and circuit
    qc = _swap_test_init(psi, phi)

    qc.h(0)

    num_qubits = int((qc.num_qubits - 1)/2)
    if idx_list == None:
        for i in range(num_qubits):
            qc.cswap(0, i+1, num_qubits+i+1)
    else:
        for i in idx_list:
            qc.cswap(0, i+1, num_qubits+i+1)

    qc.h(0)
    qc.measure(0, 0)

    return qc


def swap_overlap(psi, phi, idx_list=None, shots=1000, backend='qasm_simulator', noise=False):
    '''given two quantum circuits that represent
       psi and phi, returns the overlap between them using
       the swap test
    '''

    f = simulate_qc_with_noise if noise else simulate_qc
    qc = swap_test_QC(psi, phi, idx_list)
    counts = f(qc, shots=shots, bname=backend)
    if '1' not in counts.keys():
        return 1
    zeros, ones = counts['0'], counts['1']
    P = zeros + ones
    return 1 - 2*ones/P
