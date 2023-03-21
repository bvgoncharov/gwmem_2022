#!/bin/bash
#SBATCH --job-name=test_slurm
#SBATCH --output=/fred/oz031/mem/logs_gwmem_2022/test_slurm_20220320_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --time=15:00
#SBATCH --mem-per-cpu=2G
#SBATCH --tmp=2G
#SBATCH --array=0-0

export OMP_NUM_THREADS=1

srun singularity exec --bind "/fred/oz031/mem/gwmem_2022_container/image_content/:$HOME" python gwfish_memory_pipeline.py --outdir "../../out_gwmem_2022/" --injfile "../../pops_gwmem_2022/GWTC-like_population_20230314.hdf5" --waveform "NRHybSur3dq8" --waveform_class "gw.waveforms.LALFD_Waveform" --det "ET" --config "./detectors/gwfish_detectors.yaml" --fisher_pars "ra,dec,psi,theta_jn,luminosity_distance,mass_1,mass_2,a_1,a_2,geocent_time,phase" --td_fmin 3. --mem_sim "" --label "20230320" --num 1 --inj $SLURM_ARRAY_TASK_ID
