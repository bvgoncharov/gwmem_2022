#!/bin/bash
#SBATCH --job-name=test_slurm
#SBATCH --output=/fred/oz031/mem/logs_gwmem_2022/test_slurm_20220320_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --time=15:00
#SBATCH --mem-per-cpu=2G
#SBATCH --tmp=2G
#SBATCH --array=0-0

export OMP_NUM_THREADS=1

srun python /fred/oz031/mem/logs_gwmem_2022/gwfish_memory_pipeline.py --outdir "" --injfile "" --waveform "NRHybSur3dq8" --waveform_class "" --det "ET" --config "./gwfish_detectors.yaml" --fisher_pars "mass_1,mass_2" --td_fmin 3. --mem_sim "" --label "20230320" --num 1 --inj $SLURM_ARRAY_TASK_ID
