# Learning of discrete models of variational PDEs from data
Accompanying source code for the article

	Christian Offen, Sina Ober-Blöbaum
	Learning of discrete models of variational PDEs from data
	Status: Preprint (arXiv:2308.05082)
	
	
<a href="https://arxiv.org/abs/2308.05082">Preprint arXiv:2308.05082</a><br>
<a href="https://arxiv.org/a/offen_c_1.html">ArXiv author page</a>


# Main files - Numerical experiment with wave equation
Trains and evaluates a neural network model of a discrete Lagrangian density for a discrete wave equation.
(Also see <a href="https://github.com/Christian-Offen/LagrangianDensityML">GitHub:Christian-Offen/LagrangianDensityML</a> refering to the publication <a href="https://doi.org/10.1007/978-3-031-38271-0_57">DOI</a>.)
	
## FitDensity.jl
The scripts creates training data of a discrete field theory (discrete wave equation). Based on the training data it learns a model of discrete Lagrangian density.

## Evaluation_Trained_Model.ipynb
Jupyter notebook containing numerical experiments with a machine learned discrete density on data of the discrete wave equation. Prediction accuracy is assessed and travelling waves are detected and compared to a reference.


# Support files

## 7ptStencilFun.jl
Variational integrator for 1st order discrete field theories (2 dimensional space-time) and tools for preformance evaluation.

## SpectralTools.jl
Tools for spectral interpolation and computation of spectral derivatives on periodic spatial domains.

## TrainingData.jl
Creation of training data to be used in FitDensity.jl

## 2023-01-31_08-53-14run_data.json
Learned model of a Lagrangian density. Created by FitDensity.jl

## 2023-02-14_11-21-06run_dataTW05pert.json
Learned Fourier coefficients of travelling wave. Created by Evaluation_Trained_Model.ipynb


# MOR
Refer to the subfolder "MOR" for an experiment based on model order reduction with a latent space identified using principle component analysis.


