# gwmem_2022

Simulation of the gravitational wave memory signals for LISA and ground-based detectors (LIGO-Virgo, LIGO Voyager, Einstein Telescope, Cosmic Explorer). Model selection and parameter estimation. 

Based on the paper: 

[Inferring fundamental spacetime symmetries with gravitational-wave memory: from LISA to the Einstein Telescope](https://arxiv.org/abs/2310.10718)

*Boris Goncharov, Laura Donnay, Jan Harms (2024, PRL)*

## Environment to run the code

The first step is to build a singularity/apptainer container image. The container definition file is here: `/.def`. To build a container, run ``. Next, open container image: `singularity shell --bind "/some_new_directory_on_your_machine/:$HOME" gwmem_2022_20230321_3.sif`. A few more steps are still needed. First, move surrogate waveforms to `~/gwsurrogate_downloads` in the container. Second, this needs to be performed when inside the container:
```
pip install ChainConsumer --user
git clone git@github.com:bvgoncharov/GWFish.git
cd GWFish
git checkout development_bg
python -m pip install --user .
```
Note, `python setup.py install` did not work for me for GWFish, the package was visible but not its modules.

Now, you can git clone this repository to `/some_new_directory_on_your_machine/`, the code should work with `singularity exec`. For example:

```
singularity exec --bind "/some_new_directory_on_your_machine/:$HOME" gwmem_2022_20230321_3.sif python /home/bgonchar/gwmem_2022/gwfish_analysis/memory_strain_amplitudes.py --outdir "/home/bgonchar/out_gwmem_2022/" --injfile "/home/bgonchar/pops_gwmem_2022/pop_max_o3_bbh_only_1yr_20230606.hdf5" --waveform "NRHybSur3dq8" --waveform_class "gu.LALTD_SPH_Memory" --det "ET" --config "/home/bgonchar/gwmem_2022/gwfish_analysis/detectors/gwfish_detectors_et_10_1024Hz.yaml" --fisher_pars "ra,dec,psi,theta_jn,luminosity_distance,mass_1,mass_2,a_1,a_2,geocent_time,phase,J_E,J_J" --td_fmin 9. --f_ref 20. --mem_sim "J_E,J_J" --j_e 1.0 --j_j 1.0 --label "20230606_m" --num 100 --inj 0
```

Make sure to adjust paths, and check out other examples in `gwfish_analysis/slurm/`.

<details> 
  <summary>An alternative approach: setting up conda environment</summary>

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

</details>
