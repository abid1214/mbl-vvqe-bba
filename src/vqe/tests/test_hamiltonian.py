from .context import hamiltonian as ham

import numpy as np
import numpy.linalg as nla

def test_heisenberg1D_5qubits():
    # Checks that the heisenerg1D() and magnetic_fields() functions are
    # implemented correctly. Specifically, checks that the energy eigenvalues
    # of the Hamiltonian are correct compared to another well-tested code.
    
    num_qubits = 5

    potentials = np.array([0.1, -0.7, 5.2, -7.2, 0.8])
    
    H = ham.heisenberg1D(num_qubits) + ham.magnetic_fields(potentials)
    
    # From another well-tested code, this is what the energy eigenvalues
    # of the Hamiltonian should be.
    expected_evals = np.array([-8.24956340e+00, -7.49006471e+00, -6.92193291e+00, -6.58379999e+00,
                               -6.34938525e+00, -6.16238413e+00, -5.25646223e+00, -5.02209117e+00,
                               -2.18855213e+00, -1.23297807e+00, -1.13515612e+00, -1.09682565e+00,
                               -8.55618022e-01, -6.61424871e-03,  1.00000000e-01,  1.97744688e-01,
                               2.36041961e-01,  3.22597196e-01,  7.49290932e-01,  1.07853334e+00,
                               1.66442859e+00,  1.90000000e+00,  1.99354710e+00,  2.22909920e+00,
                               5.06732977e+00,  5.40185524e+00,  6.01873779e+00,  6.12239922e+00,
                               6.16247170e+00,  6.35323554e+00,  6.45699975e+00,  6.49711598e+00])
    
    H_mat          = H.to_matrix()
    (evals, evecs) = nla.eigh(H_mat)
    
    assert(np.allclose(evals, expected_evals, rtol=1e-8))
