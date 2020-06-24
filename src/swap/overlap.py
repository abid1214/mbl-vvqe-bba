from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

def swap_test_QC(num_qubits):
    ''' constructing algorithm from wiki
        https://en.wikipedia.org/wiki/Swap_test
        Inputs:
            num_qbits = number of quibits in psi and phi
        Return:
            QuantumCircuit object for swap test
    '''
    #initialize registers and circuit
    psi = QuantumRegister(num_qubits, 'psi')
    phi = QuantumRegister(num_qubits, 'phi')
    a   = QuantumRegister(1, 'ancilla')
    M   = ClassicalRegister(1, 'm')
    qc  = QuantumCircuit(psi, phi, a, M)

    #construct gates
    qc.h(a[0])
    for i in range(num_qubits):
        qc.cswap(a[0], psi[i], phi[i])
    qc.h(a[0])
    qc.measure(a[0], M[0])

    return qc


def ABBA_QC(num_qubits):
    '''
    This alorithm is obtained from  https://arxiv.org/pdf/1803.04114.pdf
    (see figure 6).
    '''
    #initialize registers
    psi = QuantumRegister(num_qubits, 'psi')
    phi = QuantumRegister(num_qubits, 'phi')
    M   = ClassicalRegister(2*num_qubits, 'm')
    qc  = QuantumCircuit(psi, phi, M)

    #construct gates
    for i in range(num_qubits):
        qc.cx(psi[i], phi[i])
    for i in range(num_qubits):
        qc.h(psi[i])
        qc.h(phi[i])
    for i in range(num_qubits):
        qc.measure(psi[i], M[2*i])
        qc.measure(phi[i], M[2*i+1])
    return qc
