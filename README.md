# gwmem_2022

## Requirements

Create a new conda virtual environment like this:
```
conda create -n gwmem_sxs -y -c conda-forge python=3.9
conda activate gwmem_sxs
conda install -c conda-forge gwsurrogate
conda install -c conda-forge sxs
conda install -c conda-forge matplotlib
conda install -c conda-forge bilby
```

To link the conda environment to Jupyter notebook:
```
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=gwmem_sxs
```

To load the environment: `conda activate gwmem_sxs`
