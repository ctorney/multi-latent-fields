# Estimating the abundance of a group-living species using multi-latent spatial fields

Code and data for inferring the abundance of aggregated species from aerial survey images. 

The repository includes code to generate synthetic data to test the method, survey data from the 2015 wildebeest count along with notebooks and scripts to run the analysis.

## Folder structure

### data
This directory contains the notebooks used to generate synthetic data along with subfolders for storing synthetic data and counts from the 2015 survey. The notebook `spatial_field_sim_data.ipynb` creates the distributions shown in Figure 1 of the paper, while the remaining two notebooks create the different sample designs and covariate analysis presented in the Supplementary Material.

### inla  
R script to run the inla-bru analysis. Once the simulated data has been created this script will analyze the data using inla-bru and create the population estimates shown in Fig.2C of the manuscript
  
### jolly  
Notebook for running Jolly II analysis. This notebook is run after the simulated data has been created and creates Fig.2B of the manuscript.
  
### mlgp  
Multilatent field code. Notebook for analysis of 2015 data shown in Fig3 of the manuscript. This folder also contains python scripts for running the repeated analysis with varying coverage fractions and additional files for running the model with covariates and with different sampling designs.
- `multilatent_batch.py` runs the analysis on the simulated data for creating Fig.2A of the manuscript
- `covariate_batch.py` runs the analysis on the simulated data with the inclusion of environmental covariates for creating Fig.S1 of the manuscript
- `sampling_batch.py` runs the analysis on the simulated data with varying sampling stragegies for creating Fig.S3 of the manuscript

 
### plotting  
Code for making the plots shown in the paper
  
### results
Directory for storing the outputs of the models.
