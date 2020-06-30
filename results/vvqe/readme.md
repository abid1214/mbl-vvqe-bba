# VVQE
Filenames: `W[#]_q[#]_[method]_[entanglement]_rep[#].pkl`

Data is pickled lists stored in dictionaries of

- W
- opt_params
- statevector
- E
- Var
- fidelity


To read do
```python
  import pickle
  with open(fname,'rb') as f:
    data = pickle.load(f)
```

To calculate entropy:
```python
  from qiskit.quantum_info import partial_trace,entropy
  rho = partial_trace(vec,range(num_qubits//2))
  S = entropy(rho)
  S2 = (rho.data**2).trace()
```
To load parameters into an ansatz:
```python
  ansatz = vf.sz_conserved_ansatz(num_qubits, entanglement=entanglement, reps=reps)
  params = sorted(ansatz.parameters, key=lambda p: p.name) 
  params = dict(zip(params, data[i]['opt_params']))
  newAnsatz = ansatz.assign_parameters(params)
```
