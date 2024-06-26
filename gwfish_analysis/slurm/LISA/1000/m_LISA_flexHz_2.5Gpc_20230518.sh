#!/bin/bash
#SBATCH --job-name=m_li_2.5gp_20230518
#SBATCH --output=/fred/oz031/mem/gwmem_2022_container/image_content/logs_gwmem_2022/m_LISA_flexHz_2.5Gpc_20230518_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --time=0-11
#SBATCH --mem-per-cpu=4G
#SBATCH --tmp=6G
#SBATCH --array=0-9

export OMP_NUM_THREADS=1

srun singularity exec --bind "/fred/oz031/mem/gwmem_2022_container/image_content/:$HOME" /fred/oz031/mem/gwmem_2022_container/gwmem_2022_20230321_3.sif python /home/bgonchar/gwmem_2022/gwfish_analysis/gwfish_memory_pipeline.py --outdir "/home/bgonchar/out_gwmem_2022/" --injfile "/home/bgonchar/pops_gwmem_2022/LISA_BBHs_1000_M1e6_D2-5Gpc_20230518.hdf5" --waveform "NRHybSur3dq8" --waveform_class "gu.NRSurSPH_Memory" --det "LISA" --config "/home/bgonchar/gwmem_2022/gwfish_analysis/detectors/gwfish_detectors_et_10_1024Hz.yaml" --fisher_pars "ra,dec,psi,theta_jn,luminosity_distance,mass_1,mass_2,a_1,a_2,geocent_time,phase,J_E,J_J" --td_fmin 1e-4 --f_ref 2e-4 --mem_sim "J_E,J_J" --j_e 1.0 --j_j 1.0 --label "20230518_m" --num 100 --inj $SLURM_ARRAY_TASK_ID
