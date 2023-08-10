# Learning of discrete models of variational PDEs from data
Accompanying source code for the article

	Christian Offen, Sina Ober-Blöbaum
	Learning of discrete models of variational PDEs from data
	Status: Preprint (arXiv:2308.05082)
	
	
<a href="https://arxiv.org/abs/2308.05082">Preprint arXiv:2308.05082</a><br>
<a href="https://arxiv.org/a/offen_c_1.html">ArXiv author page</a>

# Description of files for experiment with coarse mesh for discrete wave equation

## Evaluation_Trained_Model.ipynb
Jupyter notebook containing numerical experiments with a machine learned discrete density on data of the discrete wave equation. The mesh on which the model is trained is coarser than the mesh used to produce training data. This can be compared to the experiments in "../wave_equation"

## create_training_data_run.jl
produces training data and writes the file "training_data.json"

## train_subsampled_run.jl
Trains neural network model based on training data in the file "training_data.json". Model parameters are saved as "*run_param_data.json"

## train_subsampled_continuation.jl
Trains a pretrained neural network model based. Uses training data in the file "training_data.json". Reads parameters from pre-training from "*run_param_data.json". Model parameters are saved as "*run_param_data.json".

