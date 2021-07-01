#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

#SBATCH --mem=8GB
#SBATCH --time=8:00:00
#SBATCH --job-name=Trellis_Minibatch_Grid_%j

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mdd424@nyu.edu

#SBATCH --output=slurm_trellis_minibatch_%a.out

module purge

source $HOME/.bashrc

RUNDIR="$HOME/research/trellis_inference"
cd $RUNDIR

if [[ $(hostname -s) =~ ^g ]]; then nv="--nv"; fi

NUM_BATCHES=25

for (( run=0; run<$(($NUM_BATCHES + 0)); run++ )); do

    singularity exec $nv \
                --overlay $SCRATCH/environments/overlay_pytorch.ext3:ro \
                /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
                /bin/bash -c "
    source /ext3/env.sh
    python run_trellis_minibatches.py $run $(( ${SLURM_ARRAY_TASK_ID} ))
    "
done
