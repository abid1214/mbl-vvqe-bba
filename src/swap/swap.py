from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

def swap_test_circuit(n, P):
    ''' constructing algorithm from wiki
        https://en.wikipedia.org/wiki/Swap_test
        Inputs:
            n = number of quibits in psi and phi
            P = number of times algorithm should be executed
                larger P -> more accurate solution
        Return:
            QuantumCircuit object for swap test
    '''
    psi = QuantumRegister(n, 'psi')
    phi = QuantumRegister(n, 'phi')
    a  = QuantumRegister(1, 'ancilla')
    M  = ClassicalRegister(P, 'm')
    qc = QuantumCircuit(psi, phi, a, M)

    for j in range(P):
        qc.h(a[0])
        for i in range(n):
            qc.cswap(a[0], psi[i], phi[i])
        qc.h(a[0])
        qc.measure(a[0], M[j])
    return qc
