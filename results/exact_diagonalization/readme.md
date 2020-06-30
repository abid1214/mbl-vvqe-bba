# Exact Diagonalization

filenames: `ed_entropy_data_[#]_qubits.pkl`


pickled list of dictionaries:
- W
- zero_energy_vec
- reduced_dm
- energy
- renyi_entropy
- vn_entropy

==========================================  

For the spectrum data
filenames: `ed_spectrum_data_[#]_qubits.pkl.gz`

to load:
```python
  import pickle,gzip
  data = pickle.load(gzip.open("ed_spectrum_data_4_qubits.pkl.gz","rb"))
```

it contains a pickled list of lists of dicts (each list denotes a different W)
- W
- evals: eigenvalues
- evecs: eigenvectors (v_i = evecs[:,i])
- realization: disorder potential index, i.e. uses ith potential from q[#]_1000potentials.pkl

large spectrum data not uploaded due to file size

======  

`ed_entropyData_df.pkl.gz` is a pickled pandas DataFrame with

- E_order: energy eigenstate ordering 
- Energy: energy of the eigenstate 
- LA: L_A subsystem size cut 
- S1: Von Neumann entropy (1st Renyi entropy) of the L_A cut 
- S2: 2nd Renyi Entropy of the L_A cut 
- W: Disorder strength 
- realization: index of disorder in q[#]_1000potentials.pkl 
