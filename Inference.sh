# mnist_job.sh
#!/bin/bash
#BSUB -J infer      # Job name
#BSUB -q c02613                  # GPU queue gpu100 or gpuv100 or c02613
#BSUB -W 0:30                    # Walltime (2 hours)
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"       # Request 8GB of system memory
#BSUB -o ./logs/outputLogs/output_infer_%J.log            # Output file
#BSUB -e ./logs/errorLogs/error_infer_%J.log             # Error file

lscpu

# Load necessary modules
module load python3/3.10.12
module load cuda/12.1

# Activate virtual environment
source /zhome/1a/a/156609/project/path/.venv/bin/activate

# === INFERENCE ===
python3 -m Skeleton_model.Evaluation.Inference Skeleton_g10_p1_gc03_l15_w15_dice_layer3,Skeleton_g10_p1_gc03_l15_w15_dice_kernel5






# python3 -m Skeleton_model.Evaluation.Inference Skeleton_g10_p1_gc01
# python3 -m Skeleton_model.Evaluation.Inference Skeleton_g10_p1_gc05
# python3 -m Skeleton_model.Evaluation.Inference Skeleton_g10_p1_gc03
# python3 -m Skeleton_model.Evaluation.Inference Skeleton_g10_p10_gc03
# python3 -m Skeleton_model.Evaluation.Inference Skeleton_g20_p1_gc03

# # === BASELINE ===
# python3 -m Skeleton_model.Evaluation.Inference Baseline

# # === MICCAI ===
# python3 -m Skeleton_model.Evaluation.Inference Miccai_seg
# python3 -m Skeleton_model.Evaluation.Inference Miccai_skel

# # === SEGMENTATION ===
# python3 -m Skeleton_model.Evaluation.Inference Segmentation_g10_p1
# python3 -m Skeleton_model.Evaluation.Inference Segmentation_g10_p5
# python3 -m Skeleton_model.Evaluation.Inference Segmentation_g20_p1

# # === SEGMENTATION GAP CHANCE ===
# python3 -m Skeleton_model.Evaluation.Inference Segmentation_g10_p1_gs01
# python3 -m Skeleton_model.Evaluation.Inference Segmentation_g10_p1_gs05
# python3 -m Skeleton_model.Evaluation.Inference Segmentation_g10_p1_gs09

# # === SKELETON ===
# python3 -m Skeleton_model.Evaluation.Inference Skeleton_g10_p1
# python3 -m Skeleton_model.Evaluation.Inference Skeleton_g10_p5
# python3 -m Skeleton_model.Evaluation.Inference Skeleton_g20_p1