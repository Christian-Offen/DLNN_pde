# Learning of discrete Lagrangian densities from data
Accompanying source code for the conference paper

	Christian Offen, Sina Ober-Blöbaum
	Learning discrete Lagrangians for variational PDEs from data and detection of travelling waves
	In: F. Nielsen and F. Barbaresco (Eds.): GSI 2023, LNCS 14071, pp. 1–11, 2023.
	DOI: 10.1007/978-3-031-38271-0_57
 	
	Status: In press.

<a href="https://doi.org/10.1007/978-3-031-38271-0_57">DOI SpringerLink</a><br>
<a href="https://arxiv.org/abs/2302.08232">Preprint arXiv:2302.08232</a><br>
<a href="https://arxiv.org/a/offen_c_1.html">ArXiv author page</a>

# Main files
	
## FitDensity.jl
The scripts creates training data of a discrete field theory (discrete wave equation). Based on the training data it learns a model of discrete Lagrangian density.

## Evaluation_Trained_Model.ipynb
Jupyter notebook containing numerical experiments with a machine learned discrete density on data of the discrete wave equation. Prediction accuracy is assessed and travelling waves are detected and compared to a reference.


# Supporting files

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
