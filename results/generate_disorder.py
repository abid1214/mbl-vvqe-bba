import numpy as np
import pickle
np.random.seed(42)

num_qubits   = 4
realizations = 1000
potentials   = []

name = "q{}_{}potentials".format(num_qubits,realizations)

with open(name+".txt",'w') as f:
  for disorder in range(realizations):
    potential = 2.0*np.random.rand(num_qubits) - 1.0
    potentials.append(potential)
    f.write("{}\n".format(potential))

with open(name+".pkl",'wb') as f:
  pickle.dump(potentials,f)
