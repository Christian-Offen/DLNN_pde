# Learning of discrete models of variational PDEs from data
Accompanying source code for the article

	Christian Offen, Sina Ober-Blöbaum
	Learning of discrete models of variational PDEs from data
	Status: Preprint (arXiv:2308.05082)
	
	
<a href="https://arxiv.org/abs/2308.05082">Preprint arXiv:2308.05082</a><br>
<a href="https://arxiv.org/a/offen_c_1.html">ArXiv author page</a>

# Files for Schrödinger equation experiment
A neural network model of a discrete Lagrangian density is trained on data of a discrete Schrödinger equation.

## Evaluation_Trained_Model.ipynb
Jupyter notebook containing numerical experiments with the trained model. 

## FitDensity_Schroedinger.jl
Produces training data and trains a randomly initialised neural network model. 
Continuously saves optimised model pararameters to "*run_param_data.json"


## FitDensity_Schroedinger_continue1.jl
Produces training data and trains a neural network model. Parameters are initialised from "*run_param_data.json" produced by a previous run of "FitDensity_Schroedinger.jl" or "FitDensity_Schroedinger_continue1.jl".
Continuously saves optimised model pararameters to "*run_param_data.json"
