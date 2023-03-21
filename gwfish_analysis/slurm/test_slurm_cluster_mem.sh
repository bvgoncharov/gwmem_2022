#!/bin/bash
#SBATCH --job-name=test_slurm
#SBATCH --output=/fred/oz031/mem/gwmem_2022_container/image_content/logs_gwmem_2022/test_slurm_mem_20220321_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --time=15:00
#SBATCH --mem-per-cpu=2G
#SBATCH --tmp=6G
#SBATCH --array=0-0

export OMP_NUM_THREADS=1

srun singularity exec --bind "/fred/oz031/mem/gwmem_2022_container/image_content/:$HOME" /fred/oz031/mem/gwmem_2022_container/gwmem_2022_20230321_3.sif python /home/bgonchar/gwmem_2022/gwfish_analysis/gwfish_memory_pipeline.py --outdir "/home/bgonchar/out_gwmem_2022/" --injfile "/home/bgonchar/pops_gwmem_2022/GWTC-like_population_20230314.hdf5" --waveform "NRHybSur3dq8" --waveform_class "gu.LALTD_SPH_Memory" --det "ET" --config "/home/bgonchar/gwmem_2022/gwfish_analysis/detectors/gwfish_et_test.yaml" --fisher_pars "ra,dec,psi,theta_jn,luminosity_distance,mass_1,mass_2,a_1,a_2,geocent_time,phase" --td_fmin 3. --mem_sim "" --label "20230321_mem" --num 1 --inj $SLURM_ARRAY_TASK_ID
