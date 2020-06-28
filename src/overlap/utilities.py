from qiskit import Aer, execute, QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np
from math import acos

def simulate_qc(qc, shots=1000):
    '''simulates the quantum circuits, and
       returns a dictionary with the keys being
       the binary representation of the measurement
       and the values being the number of counts that
       measurement recieved. The total number of counts
       equals shots
    '''
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(qc, simulator, shots=shots)
    result = job.result()
    counts = result.get_counts(qc)
    return counts


def get_p_from_counts(counts, num_qubits):
    '''returns a vector probability distribution from
       the counts dictionary object
    '''
    p = np.zeros(2**num_qubits)
    for b in counts.keys():
        p[int(b,2)] = counts[b]
    return p/np.sum(p)


def classical_overlap(psi_vec, phi_vec):
    '''computes the overlap |<psi|phi>|^2 for two
       wavefunctions.
       psi_vec and phi_vec are vectors of amplitudes
    '''
    psi_phi = np.dot(psi_vec, np.conj(phi_vec))
    return (psi_phi*np.conj(psi_phi)).real


def psi_qc(alpha):
    '''constructs the QuantumCircuit representing
       the wavefunction
       |psi> = (|0> + e^{i alpha}|1>)/sqrt(2)
    '''
    psi = QuantumCircuit(1)
    psi.h(0)
    if alpha % 2*np.pi != 0:
        psi.rz(alpha,0)
    return psi


def measure_qc(psi):
    ''' constructs a QuantumCircuit from a
        wavefunction QuantumCircuit psi that
        has classical bits to measure
    '''
    num_qubits = psi.num_qubits
    qc = QuantumCircuit(QuantumRegister(num_qubits, 'psi'),
                        ClassicalRegister(num_qubits, 'm'))
    qc.compose(psi, list(range(num_qubits)), inplace=True)
    l = list(range(num_qubits))
    qc.measure(l,l[::-1])
    return qc


def psi_classical(alpha):
    '''constructs a vector representation of
       the wavefunction
       |psi> = (|0> + e^{i alpha}|1>)/sqrt(2)
    '''
    return np.array([1, np.exp(1j*alpha)]/np.sqrt(2))


def psi2_qc(r):
    ''' constructs the QuantumCircuit representing
        the wavefunction
        |psi> = (sqrt(1+r)(|00> + |01>) + sqrt(1-r)(|10> - |11>))/2
        where 0 <= r <= 1
        when r = 0, this is a maximally mixed state
        when r = 1, this is a pure state
    '''
    psi = QuantumCircuit(2)
    theta_over_2 = acos(np.sqrt((1+r)/2))
    psi.ry(2*theta_over_2, 0)
    psi.h(1)
    psi.cz(0,1)
    return psi


def psi2_classical(r):
    ''' constructs the vector representation of
        the wavefunction
        |psi> = (sqrt(1+r)(|00> + |01>) + sqrt(1-r)(|10> - |11>))/2
        where 0 <= r <= 1
        when r = 0, this is a maximally mixed state
        when r = 1, this is a pure state
    '''
    return np.array([np.sqrt(1+r), np.sqrt(1+r), np.sqrt(1-r), -np.sqrt(1-r)])/2


def classical_partial_overlap(psi, phi):
    '''
        constructs the overlap Tr(rhoA*sigmaA), where A is the first qubit of a
        2-qubit state psi and phi. the inputs are vectors representing the
        2-qubit states
    '''
    a, b, c, d = psi
    ac, bc, cc, dc = np.conj(psi)
    rhoA = np.array([[a*ac +b*bc, a*cc+b*dc], [ac*c+bc*d, c*cc+d*dc]])

    a, b, c, d = phi
    ac, bc, cc, dc = np.conj(phi)
    sigmaA = np.array([[a*ac +b*bc, a*cc+b*dc], [ac*c+bc*d, c*cc+d*dc]])

    return np.trace(rhoA@sigmaA)

