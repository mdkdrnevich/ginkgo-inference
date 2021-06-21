#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

#SBATCH --mem=8GB
#SBATCH --time=2:00:00
#SBATCH --job-name=Trellis_Grid_%j

#SBATCH --output=slurm_trellis_%a.out

module purge

source $HOME/.bashrc

RUNDIR="$HOME/research/trellis_inference"
cd $RUNDIR

if [[ $(hostname -s) =~ ^g ]]; then nv="--nv"; fi

singularity exec $nv \
            --overlay $SCRATCH/environments/overlay_pytorch.ext3:ro \
            /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
            /bin/bash -c "
source /ext3/env.sh
python run_trellis.py $(( ${SLURM_ARRAY_TASK_ID} ))
"
