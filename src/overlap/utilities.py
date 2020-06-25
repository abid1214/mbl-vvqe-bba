from qiskit import Aer, execute, QuantumCircuit
import numpy as np
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

def psi_classical(alpha):
    '''constructs a vector representation of
       the wavefunction
       |psi> = (|0> + e^{i alpha}|1>)/sqrt(2)
    '''
    return np.array([1, np.exp(1j*alpha)]/np.sqrt(2))
