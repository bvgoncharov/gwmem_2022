# gwmem_2022

## Setting up the environment

### On a local computer

Please follow these steps. Create a new conda virtual environment like this:
```
conda create -n gwmem_sxs -y -c conda-forge python=3.9
conda activate gwmem_sxs
conda install -c conda-forge gwsurrogate
conda install -c conda-forge sxs
conda install -c conda-forge matplotlib
conda install -c conda-forge bilby
pip install git+ssh://git@github.com/ColmTalbot/gwmemory.git
pip install git+ssh://git@github.com/jblackma/NRSur7dq2.git
```
Additionally, I work with my fork of GWFish, locally-installed with `git clone` and `python setup.py develop`.
Optional, useful: `conda install -c conda-forge ipdb`
Optional, for `gwmemory` notebook: `conda install -c conda-forge basemap`
Optional, for plotting Fisher matrix errors: `pip install git+ssh://git@github.com/bvgoncharov/normal_corner.git`

In development, gwmemory of Colm was required for jupyter notebook, gwmemory of Moritz required to test waveforms:
```
memestr
gwmemory/gwmemory_mh
```

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
Sometimes the ipython kernel will not see modules, and needs to be reinstalled: `jupyter kernelspec uninstall gwmem_sxs`.

To delete the environment (after it is deactivated): `conda env remove -n gwmem_sxs`

### On a cluster

It is best to load dependencies from the container. I created a file `gwmem_2022_20230321.sif`.
