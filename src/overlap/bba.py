from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from utilities import *

def _bba_init(psi, phi, idx_list=None):
    '''helper function for bba_QC
       initializes a quantum circuit for swap test
    '''
    num_qubits = psi.num_qubits
    num_bits = 2*num_qubits if idx_list == None else 2*len(idx_list)
    qc = QuantumCircuit(QuantumRegister(num_qubits, 'psi'),
                        QuantumRegister(num_qubits, 'phi'),
                        ClassicalRegister(num_bits, 'm'))
    qc.compose(psi, list(range(num_qubits)), inplace=True)
    qc.compose(phi, list(range(num_qubits,2*num_qubits)), inplace=True)
    return qc




def bba_QC(psi, phi, idx_list=None):
    '''
    This alorithm is obtained from  https://arxiv.org/pdf/1803.04114.pdf
    (see figure 6).
    The inputs are QuantumCircut objects representing wavefunctions psi
    and phi
    Returns a quantum circuit, where the measurements in the end
    are in the order of psi[0],phi[0], ...psi[n], phi[n]
    '''
    qc  = _bba_init(psi, phi, idx_list)
    num_qubits = int(qc.num_qubits/2)

    iter_list = list(range(num_qubits)) if idx_list == None else idx_list

    #construct gates
    [qc.cx(i, num_qubits + i) for i in iter_list]
    [qc.h(i)                  for i in iter_list]
    for i in iter_list:
        qc.measure(i,              2*i  )
        qc.measure(num_qubits + i, 2*i+1)

    return qc


def bba_overlap(psi, phi, idx_list=None, shots=1000, backend='qasm_simulator', noise=False):
    '''given two quantum circuits that represent
       psi and phi, returns the overlap between them using
       the swap test
    '''

    f = simulate_qc_with_noise if noise else simulate_qc
    num_qubits = psi.num_qubits if idx_list == None else len(idx_list)
    qc = bba_QC(psi, phi, idx_list)
    counts = f(qc, shots=shots, bname=backend)
    p_vec = get_p_from_counts(counts, 2*num_qubits)
    c_1 = np.array([1,1,1,-1])
    c_N = np.array([1,1,1,-1])
    for i in range(num_qubits -1):
        c_N = np.kron(c_N, c_1)

    return np.dot(p_vec, c_N)


