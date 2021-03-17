<img src="https://raw.githubusercontent.com/abid1214/mbl-vvqe-bba/master/coffee.png" width="250px" alt="logo" />  


# Exploring MBL via VVQE & BBA

A [Qiskit Summer Jam 2020](https://github.com/qiskit-community/qiskit-summer-jam-20) hackathon project

 👉 [Presentation](https://he-s3.s3.amazonaws.com/media/sprint/qiskit-community-summer-jam-mid-west/team/856107/6cd22b6projectpresentation3.pdf)  
 📓Notebooks: [Entanglement Entropy Plots](src/overlap/notebooks/entropy-plots.ipynb) | [BBA pure state test](src/overlap/notebooks/overlap-pure.ipynb ) | [BBA mixed state test](src/overlap/notebooks/overlap-mixed.ipynb)  
📊 Data: [ED](results/exact_diagonalization) | [VVQE](results/vvqe) | [BBA](results/overlap)   

## Physics Background

In the classical world we are used to matter thermally equilibrating with its environment, like a hot cup of coffee cooling down, but in the quantum world there are phases of matter known as many-body localized (MBL) phases that do not equilibrate, like a never-cooling cup of coffee. In this hackathon, we wanted to study MBL using quantum computers. 

## Implementation

We implement two quantum algorithms using [Qiskit](https://qiskit.org/)

* [VVQE](src/vqe/vvqe.py) - Variance Variationial Quantum Eigensolver
* Entropy Measurement [[1]](https://iopscience.iop.org/article/10.1088/1367-2630/aae94a)
  * [Swap Test](src/overlap/swap.py) 
  * [BBA](src/overlap/bba.py)

This opens a new path forward to study MBL with quantum computers!

### Requirements

This was built using Qiskit version `0.19.6`. To read and analyze some of the data we also recommend you use [pandas](https://pandas.pydata.org/) and [Jupyter notebooks](https://jupyter.org/). 

### Tests

Use `pytest` to run tests

# Team: Quarantine Qubits

* [Abid Khan](https://github.com/abid1214)
* [Ryan Levy](https://github.com/ryanlevy)
* [James Allen](https://github.com/jamesza2)
* [Eli Chertkov](https://github.com/echertkov)
* [Di Luo](https://github.com/rodin1000)

