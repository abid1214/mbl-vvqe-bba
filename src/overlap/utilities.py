from qiskit import Aer, execute, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers.aer.noise import NoiseModel
from qiskit.test.mock import *
import numpy as np
from math import acos


def simulate_qc(qc, shots=1000, bname='qasm_simulator', noise=None):
    '''simulates a quantum circuit, and
       returns a dictionary with the keys being
       the binary representation of the measurement
       and the values being the number of counts that
       measurement recieved. The total number of counts
       equals shots
    '''

    if noise == None:
        job = execute(qc, Aer.get_backend(bname), shots=shots)
    else:
        backend = noise()
        noise_model = NoiseModel.from_backend(backend)
        coupling_map = backend.configuration().coupling_map
        basis_gates = noise_model.basis_gates
        job = execute(qc, Aer.get_backend(bname),
                 shots=shots,
                 coupling_map=coupling_map,
                 basis_gates=basis_gates,
                 noise_model=noise_model)

    result = job.result()
    counts = result.get_counts(qc)
    return counts


def get_p_from_counts(counts, num_qubits):
    '''returns a vector probability distribution from
       the counts dictionary object
    '''
    p = np.zeros(2**num_qubits)
    for b in counts.keys():
        s = int(str(b)[::-1],2)
        p[s] = counts[b]
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
    qc.measure(l,l)
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


def run_vqe_overlap(qc_dict, overlap_func, shots=1000, noise=None):
    '''Takes in a dict qc_dict, where qc_dict[W] = qc_list is a dictionary
       of lists of QuantumCircuits qc_list for a given disorder strength W.
       Returns a dict of lists of second renyi entropy for each circuit
       calculated via overlap_func
    '''

    ent_dict = {}
    for W in qc_dict.keys():
        print("W={}".format(W))
        qc_list = qc_dict[W]
        o_list = [0]*len(qc_list)
        for i in range(len(qc_list)):
            psi = qc_list[i]
            n = psi.num_qubits//2
            overlap = overlap_func(psi, psi, list(range(n)), shots=shots, backend="qasm_simulator", noise=noise)
            o_list.append(-np.log2(overlap))
        ent_dict[W] = o_list
    return ent_dict
