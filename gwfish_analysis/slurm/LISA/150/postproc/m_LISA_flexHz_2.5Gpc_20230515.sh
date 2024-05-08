#!/bin/bash
#SBATCH --job-name=m_li_2.5gp_20230515
#SBATCH --output=/fred/oz031/mem/gwmem_2022_container/image_content/logs_gwmem_2022/m_LISA_flexHz_2.5Gpc_20230515_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --time=0-1
#SBATCH --mem-per-cpu=1G
#SBATCH --tmp=6G
#SBATCH --array=0-0

export OMP_NUM_THREADS=1

srun singularity exec --bind "/fred/oz031/mem/gwmem_2022_container/image_content/:$HOME" /fred/oz031/mem/gwmem_2022_container/gwmem_2022_20230321_3.sif python /home/bgonchar/gwmem_2022/gwfish_analysis/gwfish_memory_postprocessing.py --outdir "/home/bgonchar/out_gwmem_2022/" --injfile "/home/bgonchar/pops_gwmem_2022/LISA_BBHs_150_M1e6_D2-5Gpc_20230515.hdf5" --waveform "NRHybSur3dq8" --waveform_class "gu.NRSurSPH_Memory" --det "LISA" --config "/home/bgonchar/gwmem_2022/gwfish_analysis/detectors/gwfish_detectors_et_10_1024Hz.yaml" --fisher_pars "ra,dec,psi,theta_jn,luminosity_distance,mass_1,mass_2,a_1,a_2,geocent_time,phase,J_E,J_J" --td_fmin 1e-4 --f_ref 2e-4 --mem_sim "J_E,J_J" --j_e 1.0 --j_j 1.0 --label "20230515_m" --num 150 
