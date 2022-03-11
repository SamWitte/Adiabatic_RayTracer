#!/bin/bash
#SBATCH --time=100:00:00
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
echo This job was running on:
hostname
#export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
MassA=1.0e-6
B0=1e14
ThetaM=0.0
rotW=1.0
Trajs=900

declare -i memPerjob
memPerjob=$((SLURM_MEM_PER_NODE/SLURM_NTASKS))
echo $memPerjob
echo $SLURM_NTASKS
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

for ((i = 0; i < $SLURM_NTASKS ; i++)); do
#     srun --ntasks=1 --exclusive --mem=$memPerjob julia Gen_Samples_GR_Server.jl --MassA $MassA --B0 $B0 --ThetaM $ThetaM --rotW $rotW --Nts $Trajs --ftag $i --run_RT 1 & 
    srun --ntasks=1 --exclusive --mem=$memPerjob julia Gen_Samples_GR_Server.jl --MassA $MassA --B0 $B0 --ThetaM $ThetaM --rotW $rotW --Nts $Trajs --ftag $i --run_RT 1 &      
done
wait

srun --ntasks=1 --exclusive --mem=$memPerjob julia  Gen_Samples_GR_Server.jl --MassA $MassA --B0 $B0 --ThetaM $ThetaM --rotW $rotW --Nts $Trajs --run_RT 0 --run_Combine 1 --side_runs $SLURM_NTASKS
