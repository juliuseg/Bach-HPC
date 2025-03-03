# mnist_job.sh
#!/bin/bash
#BSUB -J Bach_training           # Job name
#BSUB -q gpua100                   # 
#BSUB -gpu "num=1"                # Request 1 GPU in exclusive mode
#BSUB -n 4                        # Request 4 CPU cores (required)
#BSUB -R "span[hosts=1]"          # Ensure resources are on a single node
#BSUB -W 72:00                    # Walltime (72 hours)
#BSUB -R "rusage[mem=32768]"       # Request 8GB of system memory
#BSUB -o ./logs/outputLogs/output_%J.log            # Output file
#BSUB -e ./logs/errorLogs/error_%J.log             # Error file

lscpu

# Load necessary modules
module load python3/3.10.12
module load cuda/12.1

# Activate virtual environment
source /zhome/1a/a/156609/project/path/.venv/bin/activate

# Run the PyTorch training script
python3 -m Skeleton_model.Train
