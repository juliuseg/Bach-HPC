#!/bin/bash
#BSUB -J email_test
#BSUB -q c02613
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -W 00:05
#BSUB -R "rusage[mem=2GB]"
#BSUB -R "span[hosts=1]"          # Ensure resources are on a single node
#BSUB -o email_test_%J.out
#BSUB -e email_test_%J.err

module load python3/3.10.12

lscpu
nvidia-smi

sleep 5

echo "Done."
