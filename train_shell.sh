# mnist_job.sh
#!/bin/bash
#BSUB -J Bach_train               # Job name
#BSUB -q c02613                  # c02613 or gpua100 or gpuv100
#BSUB -gpu "num=1"                # Request 1 GPU in exclusive mode
#BSUB -n 4                        # Request 4 CPU cores (required)
#BSUB -R "span[hosts=1]"          # Ensure resources are on a single node
#BSUB -W 00:30                    # Walltime (72 hours)
#BSUB -R "rusage[mem=32GB]"       # Request 8GB of system memory
#BSUB -o ./logs/outputLogs/bach_train_%J.log            # Output file
#BSUB -e ./logs/errorLogs/bach_train_%J.log             # Error file
#BSUB -B                 # Send email when the job begins
#BSUB -N                 # Send email when the job ends

lscpu

module load python3/3.10.12
module load cuda/12.1

source /zhome/1a/a/156609/project/path/.venv/bin/activate

notify_done() {
    msg="$1"
    bsub -q hpc -u s204427@dtu.dk -N -J notify_done \
         -o /dev/null -e /dev/null \
         /bin/bash -c "echo 'DONE: $msg at \$(date)'"
}


# Baseline
echo "Training on: gapsize 10, gap_chance 0.3, num_lines 15, wobble 1.5, loss function: dice"
python3 -m Skeleton_model.Train 10 1 0.3 15 1.5 "dice" 32,64 2 3 skeleton_shallow
notify_done "Training on: gapsize 10, gap_chance 0.3, num_lines 15, wobble 1.5, loss function: dice"

# # Vary gap_size
# echo "Training on: gapsize 5, gap_chance 0.3, num_lines 15, wobble 1.5, loss function: dice"
# python3 -m Skeleton_model.Train 5 0 0.3 15 1.5 "dice" 32,64,128 2,2 3 segmentation_decreased_gap_size
# notify_done "Training on: gapsize 5, gap_chance 0.3, num_lines 15, wobble 1.5, loss function: dice"

# echo "Training on: gapsize 20, gap_chance 0.3, num_lines 15, wobble 1.5, loss function: dice"
# python3 -m Skeleton_model.Train 20 0 0.3 15 1.5 "dice" 32,64,128 2,2 3 segmentation_increased_gap_size
# notify_done "Training on: gapsize 20, gap_chance 0.3, num_lines 15, wobble 1.5, loss function: dice"

# # Vary gap_chance
# echo "Training on: gapsize 10, gap_chance 0.1, num_lines 15, wobble 1.5, loss function: dice"
# python3 -m Skeleton_model.Train 10 0 0.1 15 1.5 "dice" 32,64,128 2,2 3 segmentation_decreased_gap_chance
# notify_done "Training on: gapsize 10, gap_chance 0.1, num_lines 15, wobble 1.5, loss function: dice"

# echo "Training on: gapsize 10, gap_chance 0.5, num_lines 15, wobble 1.5, loss function: dice"
# python3 -m Skeleton_model.Train 10 0 0.5 15 1.5 "dice" 32,64,128 2,2 3 segmentation_increased_gap_chance
# notify_done "Training on: gapsize 10, gap_chance 0.5, num_lines 15, wobble 1.5, loss function: dice"

# # Vary num_lines
# echo "Training on: gapsize 10, gap_chance 0.3, num_lines 7, wobble 1.5, loss function: dice"
# python3 -m Skeleton_model.Train 10 0 0.3 7 1.5 "dice" 32,64,128 2,2 3 segmentation_decreased_num_lines
# notify_done "Training on: gapsize 10, gap_chance 0.3, num_lines 7, wobble 1.5, loss function: dice"

# echo "Training on: gapsize 10, gap_chance 0.3, num_lines 30, wobble 1.5, loss function: dice"
# python3 -m Skeleton_model.Train 10 0 0.3 30 1.5 "dice" 32,64,128 2,2 3 segmentation_increased_num_lines
# notify_done "Training on: gapsize 10, gap_chance 0.3, num_lines 30, wobble 1.5, loss function: dice"

# # Vary wobble
# echo "Training on: gapsize 10, gap_chance 0.3, num_lines 15, wobble 1.0, loss function: dice"
# python3 -m Skeleton_model.Train 10 0 0.3 15 1.0 "dice" 32,64,128 2,2 3 segmentation_wobble_10
# notify_done "Training on: gapsize 10, gap_chance 0.3, num_lines 15, wobble 1.0, loss function: dice"

# echo "Training on: gapsize 10, gap_chance 0.3, num_lines 15, wobble 2.0, loss function: dice"
# python3 -m Skeleton_model.Train 10 0 0.3 15 2.0 "dice" 32,64,128 2,2 3 segmentation_wobble_20
# notify_done "Training on: gapsize 10, gap_chance 0.3, num_lines 15, wobble 2.0, loss function: dice"

# echo "Training on: gapsize 10, gap_chance 0.3, num_lines 15, wobble 3.0, loss function: dice"
# python3 -m Skeleton_model.Train 10 0 0.3 15 3.0 "dice" 32,64,128 2,2 3 segmentation_wobble_30
# notify_done "Training on: gapsize 10, gap_chance 0.3, num_lines 15, wobble 3.0, loss function: dice"




# (1) gap_size, (2) skeleton(bool: 0,1), (3) gap_chance, (4) num_lines, (5) wobble, (6) loss function

# echo "Training on: deeper model"
# python3 -m Skeleton_model.Train 10 1 0.3 15 1.5 "dice" 32,64,128,256 2,2,2 3 "deeper_model"
# notify_done "Training on: deeper model"

# echo "Training on: increased kernel size"
# python3 -m Skeleton_model.Train 10 1 0.3 15 1.5 "dice" 32,64,128 2,2 5 "kernel_5_model"
# notify_done "increased kernel size"

# # Vary loss function between "conn" or "dice" or "focal"
# echo "Training on: gapsize 10, gap_chance 0.3, num_lines 15, wobble 1.5, loss function: conn"
# python3 -m Skeleton_model.Train 10 1 0.3 15 1.5 "conn"
# notify_done "Training on: gapsize 10, gap_chance 0.3, num_lines 15, wobble 1.5, loss function: conn"

# echo "Training on: gapsize 10, gap_chance 0.3, num_lines 15, wobble 1.5, loss function: focal"
# python3 -m Skeleton_model.Train 10 1 0.3 15 1.5 "focal"
# notify_done "Training on: gapsize 10, gap_chance 0.3, num_lines 15, wobble 1.5, loss function: focal"

