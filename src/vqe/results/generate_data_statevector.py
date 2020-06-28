import numpy as np
from context import hamiltonian,variational_form,vvqe
from hamiltonian import *
from variational_form import sz_conserved_ansatz
from qiskit import *
from qiskit.aqua.components.optimizers import *
from qiskit.aqua.algorithms import VQE
from vvqe import VVQE
import time
import datetime
import pickle
import sys

np.random.seed(42)
entanglement    = "sca" #"full" # ansatz: entanglement type
spindn_cluster  = "balanced"    # ansatz: initial starting Sz configuration
reps            = 2             # ansatz: reps of form

num_qubits      = 4             # number of qubits
num_trials      = 100           # total # of trials
start           = 0             # first trial #

W = int(sys.argv[1])

# open saved potential files
# using the pickle instead of text in order to prevent precision problems

with open("q{}_1000potentials.pkl".format(num_qubits),'rb') as f:
  allPotentials = pickle.load(f)
for potential in allPotentials:
  assert len(potential) == num_qubits

print("Start W={} {}".format(W,datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
data = []
name = "W%s_q%s_VVQE_SLSQP_%s_rep%s"%(W,num_qubits,entanglement,reps)
with open(name+'.txt','w') as f:
  for trial in range(start,start+num_trials):
    potentials = allPotentials[trial]
    f.write("Potentials={}\n".format(potentials))
    # Hamiltonian H
    H         = heisenberg1D(num_qubits) + W * magnetic_fields(potentials)
    H_mat     = H.to_matrix()
    # H^2
    H_squared     = square(H)
    H_squared_mat = H_squared.to_matrix()

    ## Eigenvalue of H 
    (evals, evecs)   = np.linalg.eigh(H_mat)
    ## Eigenvalue of H^2 
    (evals2, evecs2) = np.linalg.eigh(H_squared_mat)
    ## Check that eigenvalues of H and H^2 match.
    assert(np.allclose(np.sort(np.abs(evals)**2.0), evals2))

    optimizer1 = SLSQP(maxiter=100) #SLSQP(maxiter=500,disp=True)
    optimizer2 = SLSQP(maxiter=500) #SLSQP(maxiter=500,disp=True)

    ansatz = sz_conserved_ansatz(num_qubits, entanglement=entanglement, spindn_cluster = spindn_cluster, seed = 99999,reps=reps)

    f.write(("Ansatz = {} {} {}\n".format(entanglement,spindn_cluster,reps)))
    f.write("Ham=Hsquared\n")
    f.write("{}\n".format(optimizer1.setting))
    f.flush()

    t0 = time.process_time()

    initial_point = (np.random.rand(ansatz.num_parameters)-0.5)*4*np.pi
    vqe_q         = VQE(H_squared, ansatz, optimizer1,initial_point=initial_point,include_custom=True)
    backend       = Aer.get_backend("statevector_simulator")

    vqe_results = vqe_q.run(backend)

    # now pass to VVQE 
    f.write("Ham=H\n")
    f.write("{}\n".format(optimizer2.setting))
    f.flush()
    vvqe_q = VVQE(H,ansatz,optimizer2,
        initial_point=vqe_results['optimal_point'],include_custom=True)
    vvqe_results = vvqe_q.run(backend)

    t1 = time.process_time()
    f.write("Total_time={}\n".format(t1-t0))
    for x in vvqe_results:
      f.write("{} {}\n".format(x,vvqe_results[x]))
    vec = vvqe_results['eigenstate']
    energy   = (vec.conj()@H_mat@vec).real
    variance = (vec.conj()@H_squared_mat@vec-energy**2).real
    f.write("energy="+str(energy)+"\n")
    f.write("variance="+str(variance)+"\n")
    fidelity=np.max(np.abs(evecs.T@vec))
    f.write("Fid="+str(fidelity)+"\n")
    f.flush()

    # accumulate data for pickle
    realization_data = {'W':W,'opt_params':vvqe_results['optimal_point'],
                        'statevector':vec,
                        'E':energy,'Var':variance,'fidelity':fidelity}
    data.append(realization_data)
# dump data to pickle
with open(name+".pkl",'wb') as f:
  pickle.dump(data,f)
print("Stop W={} {}".format(W,datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
