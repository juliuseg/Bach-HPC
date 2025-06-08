# mnist_job.sh
#!/bin/bash
#BSUB -J eval      # Job name
#BSUB -q hpc                  # GPU queue
#BSUB -W 24:00                    # Walltime (2 hours)
#BSUB -n 24
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"       # Request 8GB of system memory
#BSUB -o ./logs/outputLogs/output_eval_%J.log            # Output file
#BSUB -e ./logs/errorLogs/error_eval_%J.log             # Error file

lscpu

# Load necessary modules
module load python3/3.10.12
module load cuda/12.1

# Activate virtual environment
source /zhome/1a/a/156609/project/path/.venv/bin/activate

# notify_done() {
#     msg="$1"
#     bsub -q hpc -u s204427@dtu.dk -N -J notify_done \
#          -o /dev/null -e /dev/null \
#          /bin/bash -c "echo 'DONE: $msg at \$(date)'"
# }

echo "Evaluating: Skeleton_g10_p1_gc03_l15_w15_dice_layer1"
python3 -m Skeleton_model.Evaluation.Evaluate_Models Skeleton_g10_p1_gc03_l15_w15_dice_layer1

echo "Evaluating: Segmentation_g10_p1_gc03_l15_w15_dice"
python3 -m Skeleton_model.Evaluation.Evaluate_Models Segmentation_g10_p1_gc03_l15_w15_dice

echo "Evaluating: Segmentation_g5_p1_gc03_l15_w15_dice"
python3 -m Skeleton_model.Evaluation.Evaluate_Models Segmentation_g5_p1_gc03_l15_w15_dice

echo "Evaluating: Segmentation_g20_p1_gc03_l15_w15_dice"
python3 -m Skeleton_model.Evaluation.Evaluate_Models Segmentation_g20_p1_gc03_l15_w15_dice

echo "Evaluating: Segmentation_g10_p1_gc01_l15_w15_dice"
python3 -m Skeleton_model.Evaluation.Evaluate_Models Segmentation_g10_p1_gc01_l15_w15_dice

echo "Evaluating: Segmentation_g10_p1_gc05_l15_w15_dice"
python3 -m Skeleton_model.Evaluation.Evaluate_Models Segmentation_g10_p1_gc05_l15_w15_dice

echo "Evaluating: Segmentation_g10_p1_gc03_l7_w15_dice"
python3 -m Skeleton_model.Evaluation.Evaluate_Models Segmentation_g10_p1_gc03_l7_w15_dice

echo "Evaluating: Segmentation_g10_p1_gc03_l30_w15_dice"
python3 -m Skeleton_model.Evaluation.Evaluate_Models Segmentation_g10_p1_gc03_l30_w15_dice

echo "Evaluating: Segmentation_g10_p1_gc03_l15_w10_dice"
python3 -m Skeleton_model.Evaluation.Evaluate_Models Segmentation_g10_p1_gc03_l15_w10_dice

echo "Evaluating: Segmentation_g10_p1_gc03_l15_w20_dice"
python3 -m Skeleton_model.Evaluation.Evaluate_Models Segmentation_g10_p1_gc03_l15_w20_dice

echo "Evaluating: Segmentation_g10_p1_gc03_l15_w30_dice"
python3 -m Skeleton_model.Evaluation.Evaluate_Models Segmentation_g10_p1_gc03_l15_w30_dice






# echo "Evaluating: Baseline_5"
# python3 -m Skeleton_model.Evaluation.Evaluate_Models Baseline_5
# # notify_done "Eval done: Baseline_5"


# echo "Evaluating: Baseline_20"
# python3 -m Skeleton_model.Evaluation.Evaluate_Models Baseline_20
# # notify_done "Eval done: Baseline_20"


# echo "conloss: Skeleton_g10_p1_gc03_l15_w15_conloss"
# python3 -m Skeleton_model.Evaluation.Evaluate_Models Skeleton_g10_p1_gc03_l15_w15_conloss 24 "yes"
# notify_done "Eval done: Skeleton_g10_p1_gc03_l15_w15_conloss"

# echo "One more layer: Skeleton_g10_p1_gc03_l15_w15_dice_layer3"
# python3 -m Skeleton_model.Evaluation.Evaluate_Models Skeleton_g10_p1_gc03_l15_w15_dice_layer3
# notify_done "Eval done: Skeleton_g10_p1_gc03_l15_w15_dice_layer3"

# echo "kernel=5: Skeleton_g10_p1_gc03_l15_w15_dice_kernel5"
# python3 -m Skeleton_model.Evaluation.Evaluate_Models Skeleton_g10_p1_gc03_l15_w15_dice_kernel5
# notify_done "Eval done: Skeleton_g10_p1_gc03_l15_w15_dice_kernel5"


# echo "Default parameters: Skeleton_g10_p1_gc03_l15_w15_dice"
# python3 -m Skeleton_model.Evaluation.Evaluate_Models Skeleton_g10_p1_gc03_l15_w15_dice
# notify_done "Eval done: Skeleton_g10_p1_gc03_l15_w15_dice"

# echo "Lower gap size: Skeleton_g5_p1_gc03_l15_w15_dice"
# python3 -m Skeleton_model.Evaluation.Evaluate_Models Skeleton_g5_p1_gc03_l15_w15_dice
# notify_done "Eval done: Skeleton_g5_p1_gc03_l15_w15_dice"

# echo "Higher gap size: Skeleton_g20_p1_gc03_l15_w15_dice"
# python3 -m Skeleton_model.Evaluation.Evaluate_Models Skeleton_g20_p1_gc03_l15_w15_dice
# notify_done "Eval done: Skeleton_g20_p1_gc03_l15_w15_dice"

# echo "Lower gap chance: Skeleton_g10_p1_gc01_l15_w15_dice"
# python3 -m Skeleton_model.Evaluation.Evaluate_Models Skeleton_g10_p1_gc01_l15_w15_dice
# notify_done "Eval done: Skeleton_g10_p1_gc01_l15_w15_dice"

# echo "Higher gap chance: Skeleton_g10_p1_gc05_l15_w15_dice"
# python3 -m Skeleton_model.Evaluation.Evaluate_Models Skeleton_g10_p1_gc05_l15_w15_dice
# notify_done "Eval done: Skeleton_g10_p1_gc05_l15_w15_dice"

# echo "Lower number of lines: Skeleton_g10_p1_gc03_l7_w15_dice"
# python3 -m Skeleton_model.Evaluation.Evaluate_Models Skeleton_g10_p1_gc03_l7_w15_dice
# notify_done "Eval done: Skeleton_g10_p1_gc03_l7_w15_dice"

# echo "Higher number of lines: Skeleton_g10_p1_gc03_l30_w15_dice"
# python3 -m Skeleton_model.Evaluation.Evaluate_Models Skeleton_g10_p1_gc03_l30_w15_dice
# notify_done "Eval done: Skeleton_g10_p1_gc03_l30_w15_dice"

# echo "Wobble of 1.0: Skeleton_g10_p1_gc03_l15_w10_dice"
# python3 -m Skeleton_model.Evaluation.Evaluate_Models Skeleton_g10_p1_gc03_l15_w10_dice
# notify_done "Eval done: Skeleton_g10_p1_gc03_l15_w10_dice"

# echo "Wobble of 2.0: Skeleton_g10_p1_gc03_l15_w20_dice"
# python3 -m Skeleton_model.Evaluation.Evaluate_Models Skeleton_g10_p1_gc03_l15_w20_dice
# notify_done "Eval done: Skeleton_g10_p1_gc03_l15_w20_dice"

# echo "Wobble of 3.0: Skeleton_g10_p1_gc03_l15_w30_dice"
# python3 -m Skeleton_model.Evaluation.Evaluate_Models Skeleton_g10_p1_gc03_l15_w30_dice
# notify_done "Eval done: Skeleton_g10_p1_gc03_l15_w30_dice"

# echo "Loss function: ConLoss: Skeleton_g10_p1_gc03_l15_w15_conloss"
# python3 -m Skeleton_model.Evaluation.Evaluate_Models Skeleton_g10_p1_gc03_l15_w15_conloss
# notify_done "Eval done: Skeleton_g10_p1_gc03_l15_w15_conloss"

# echo "Loss function: FocalLoss: Skeleton_g10_p1_gc03_l15_w15_focal"
# python3 -m Skeleton_model.Evaluation.Evaluate_Models Skeleton_g10_p1_gc03_l15_w15_focal
# notify_done "Eval done: Skeleton_g10_p1_gc03_l15_w15_focal"




# python3 -m Skeleton_model.Evaluation.Evaluate_Models Skeleton_g10_p1_gc01
# python3 -m Skeleton_model.Evaluation.Evaluate_Models Skeleton_g10_p1_gc05
# python3 -m Skeleton_model.Evaluation.Evaluate_Models Skeleton_g10_p1_gc03
# python3 -m Skeleton_model.Evaluation.Evaluate_Models Skeleton_g10_p10_gc03
# python3 -m Skeleton_model.Evaluation.Evaluate_Models Skeleton_g20_p1_gc03

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
