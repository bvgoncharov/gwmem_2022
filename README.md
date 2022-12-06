# gwmem_2022

## Setting up the environment

Please follow these steps. Create a new conda virtual environment like this:
```
conda create -n gwmem_sxs -y -c conda-forge python=3.9
conda activate gwmem_sxs
conda install -c conda-forge gwsurrogate
conda install -c conda-forge sxs
conda install -c conda-forge matplotlib
conda install -c conda-forge bilby
```
Optional, useful: `conda install -c conda-forge ipdb`

After installation, deactivate the environment and activate it again:
```
conda deactivate
conda activate gwmem_sxs
```
This is necessary for LAL packages, otherwise they might not be visible.
This also helps in case of the following error: `ModuleNotFoundError: No module named 'lal'`.

Link the conda environment to Jupyter notebook:
```
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=gwmem_sxs
```
Limitations:
 - Bilby will not work with NRSur7dq4 in Jupyter notebook, only in the command line.

To delete the environment (after it is deactivated): `conda env remove -n gwmem_sxs`
