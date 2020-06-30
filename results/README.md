# VVQE Results

This folder contains results for both VVQE optimization as well as data obtained from exactly solving the different disordered Hamiltonians

## Disorder Realizations

We fix the same disorder patterns (named `potentials` in the code) for all strengths `W`. Listed in this folder are those files

`q[#]_1000potentials.pkl/txt`

which contain a text or pickled file of 1000 potentials of `[#]` qubits.

## Folder Structure
 - [exact_diagonalization](exact_diagonalization)
    - exact results to test against 
    - (some files are too large for github, email for more)
 - vvqe 
    - Results of optimizing variational circuits using VVQE
 - overlap
    - Plots and results of measuring entropy of the variational circuits

