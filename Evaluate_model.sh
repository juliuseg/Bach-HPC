# mnist_job.sh
#!/bin/bash
#BSUB -J eval      # Job name
#BSUB -q hpc                  # GPU queue
#BSUB -W 24:00                    # Walltime (2 hours)
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"       # Request 8GB of system memory
#BSUB -o ./logs/outputLogs/output_eval_%J.log            # Output file
#BSUB -e ./logs/errorLogs/error_eval_%J.log             # Error file

lscpu

# Load necessary modules
module load python3/3.10.12
module load cuda/12.1

# Activate virtual environment
source /zhome/1a/a/156609/project/path/.venv/bin/activate

python3 -m Skeleton_model.Evaluation.Evaluate_Models Skeleton_g10_p1_gc01
python3 -m Skeleton_model.Evaluation.Evaluate_Models Skeleton_g10_p1_gc05
python3 -m Skeleton_model.Evaluation.Evaluate_Models Skeleton_g10_p1_gc03
python3 -m Skeleton_model.Evaluation.Evaluate_Models Skeleton_g10_p10_gc03
python3 -m Skeleton_model.Evaluation.Evaluate_Models Skeleton_g20_p1_gc03

# # === BASELINE ===
# python3 -m Skeleton_model.Evaluation.Evaluate_Models Baseline

# # === MICCAI ===
# python3 -m Skeleton_model.Evaluation.Evaluate_Models Miccai_seg
# python3 -m Skeleton_model.Evaluation.Evaluate_Models Miccai_skel

# # === SEGMENTATION ===
# python3 -m Skeleton_model.Evaluation.Evaluate_Models Segmentation_g10_p1
# python3 -m Skeleton_model.Evaluation.Evaluate_Models Segmentation_g10_p5
# python3 -m Skeleton_model.Evaluation.Evaluate_Models Segmentation_g20_p1

# # === SEGMENTATION GAP CHANCE ===
# python3 -m Skeleton_model.Evaluation.Evaluate_Models Segmentation_g10_p1_gs01
# python3 -m Skeleton_model.Evaluation.Evaluate_Models Segmentation_g10_p1_gs05
# python3 -m Skeleton_model.Evaluation.Evaluate_Models Segmentation_g10_p1_gs09

# # === SKELETON ===
# python3 -m Skeleton_model.Evaluation.Evaluate_Models Skeleton_g10_p1
# python3 -m Skeleton_model.Evaluation.Evaluate_Models Skeleton_g10_p5
# python3 -m Skeleton_model.Evaluation.Evaluate_Models Skeleton_g20_p1
