#!/bin/bash
#SBATCH --job-name=jejj_maxo3_20230516
#SBATCH --output=/fred/oz031/mem/gwmem_2022_container/image_content/logs_gwmem_2022/jejj_maxo3_10_1024Hz_9_20230516_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --time=0-17
#SBATCH --mem-per-cpu=5G
#SBATCH --tmp=8G
#SBATCH --array=0-99

export OMP_NUM_THREADS=1

srun singularity exec --bind "/fred/oz031/mem/gwmem_2022_container/image_content/:$HOME" /fred/oz031/mem/gwmem_2022_container/gwmem_2022_20230321_3.sif python /home/bgonchar/gwmem_2022/gwfish_analysis/memory_strain_amplitudes.py --outdir "/home/bgonchar/out_gwmem_2022/" --injfile "/home/bgonchar/pops_gwmem_2022/pop_max_o3_bbh_only_20230504.hdf5" --waveform "NRHybSur3dq8" --waveform_class "gu.LALTD_SPH_Memory" --det "ET" --config "/home/bgonchar/gwmem_2022/gwfish_analysis/detectors/gwfish_detectors_et_10_1024Hz.yaml" --fisher_pars "ra,dec,psi,theta_jn,luminosity_distance,mass_1,mass_2,a_1,a_2,geocent_time,phase,J_E,J_J" --td_fmin 9. --f_ref 20. --mem_sim "J_E,J_J" --j_e 1.0 --j_j 1.0 --label "20230505_m" --num 100 --inj $SLURM_ARRAY_TASK_ID
