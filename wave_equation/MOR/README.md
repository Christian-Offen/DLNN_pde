# Learning of discrete models of variational PDEs from data
Accompanying source code for the article

	Christian Offen, Sina Ober-Blöbaum
	Learning of discrete models of variational PDEs from data
	Status: Preprint (arXiv:2308.05082)
	
	
<a href="https://arxiv.org/abs/2308.05082">Preprint arXiv:2308.05082</a><br>
<a href="https://arxiv.org/a/offen_c_1.html">ArXiv author page</a>

Experiment based on model order reduction with a latent space identified using principle component analysis (PCA). The neural network models a discrete Lagrangian on a latent space. The latent space is identified using PCA.

# main files

## LinearMOR_Evaluation.ipynb
Shows numerical experiments to evaluate the trained neural network model. 

## LinearMOR_run.jl
Trains model based on observational data "2023-01-31_08-53-14run_data.json" produced in the experiment of parental folder "../". Parameters are continuously saved in "*run_WaveROM_param_data.json".




