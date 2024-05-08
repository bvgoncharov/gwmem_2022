#!/bin/bash
#SBATCH --job-name=m_maxo3_20230608
#SBATCH --output=/fred/oz031/mem/gwmem_2022_container/image_content/logs_gwmem_2022/m_maxo3_ET_10_1024Hz_9_postproc_noise_20230609_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --time=0-2
#SBATCH --mem-per-cpu=5G
#SBATCH --tmp=8G
#SBATCH --array=35-49

export OMP_NUM_THREADS=1

srun singularity exec --bind "/fred/oz031/mem/gwmem_2022_container/image_content/:$HOME" /fred/oz031/mem/gwmem_2022_container/gwmem_2022_20230321_3.sif python /home/bgonchar/gwmem_2022/gwfish_analysis/gwfish_memory_postprocessing.py --outdir "/home/bgonchar/out_gwmem_2022/" --injfile "/home/bgonchar/pops_gwmem_2022/pop_max_o3_bbh_only_1yr_20230606_sorted_snrjj.hdf5" --waveform "NRHybSur3dq8" --waveform_class "gu.LALTD_SPH_Memory" --det "ET" --config "/home/bgonchar/gwmem_2022/gwfish_analysis/detectors/gwfish_detectors_et_10_1024Hz.yaml" --fisher_pars "ra,dec,psi,theta_jn,luminosity_distance,mass_1,mass_2,a_1,a_2,geocent_time,phase,J_E,J_J" --td_fmin 9. --f_ref 20. --mem_sim "J_E,J_J" --j_e 1.0 --j_j 1.0 --label "20230608_m" --num 200 --noise 1 --svd_threshold 1e-20 --randomize_mem_pe 1 --inj $SLURM_ARRAY_TASK_ID
