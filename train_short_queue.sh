#!/bin/bash
#BSUB -J Bach_Train_short               # Job name
#BSUB -q hpc                  # c02613 or gpua100 or gpuv100
#BSUB -gpu "num=1"                # Request 1 GPU in exclusive mode
#BSUB -n 4                        # Request 4 CPU cores (required)
#BSUB -R "span[hosts=1]"          # Ensure resources are on a single node
#BSUB -W 24:00                    # Walltime (72 hours)
#BSUB -R "rusage[mem=1GB]"       # Request 8GB of system memory
#BSUB -o ./logs/outputLogs/bach_train_short_%J.log            # Output file
#BSUB -e ./logs/errorLogs/bach_train_short_%J.log             # Error file


# General job configuration
QUEUE="c02613"
WALLTIME="00:30"
CORES=4
GPU="num=1"
MEMORY="32GB"
VENV_PATH="/zhome/1a/a/156609/project/path/.venv/bin/activate"
LOG_DIR="./logs"

# Make sure log directories exist
mkdir -p $LOG_DIR/outputLogs
mkdir -p $LOG_DIR/errorLogs

submit_job() {
  GAP_SIZE=$1
  SKELETON=$2
  GAP_CHANCE=$3
  NUM_LINES=$4
  WOBBLE=$5
  LOSS=$6
  CHANNELS=$7
  STRIDES=$8
  KERNEL=$9
  NAME=${10}

  JOB_SCRIPT=$(mktemp)
  cat <<EOF > "$JOB_SCRIPT"
#!/bin/bash
#BSUB -J $NAME
#BSUB -q $QUEUE
#BSUB -gpu "$GPU"
#BSUB -n $CORES
#BSUB -R "span[hosts=1]"
#BSUB -W $WALLTIME
#BSUB -R "rusage[mem=$MEMORY]"
#BSUB -o $LOG_DIR/outputLogs/${NAME}_%J.log
#BSUB -e $LOG_DIR/errorLogs/${NAME}_%J.log
#BSUB -B
#BSUB -N
#BSUB -u s204427@dtu.dk

module load python3/3.10.12
module load cuda/12.1
source $VENV_PATH

echo "Training $NAME"
python3 -m Skeleton_model.Train $GAP_SIZE $SKELETON $GAP_CHANCE $NUM_LINES $WOBBLE "$LOSS" $CHANNELS $STRIDES $KERNEL $NAME
EOF

  bsub < "$JOB_SCRIPT"
  rm "$JOB_SCRIPT"
}

# === Submit all jobs ===

submit_job 10 0 0.3 15 1.5 "dice" 32,64,128 2,2 3 segmentation_default
submit_job 5 0 0.3 15 1.5 "dice" 32,64,128 2,2 3 segmentation_decreased_gap_size
submit_job 20 0 0.3 15 1.5 "dice" 32,64,128 2,2 3 segmentation_increased_gap_size
submit_job 10 0 0.1 15 1.5 "dice" 32,64,128 2,2 3 segmentation_decreased_gap_chance
submit_job 10 0 0.5 15 1.5 "dice" 32,64,128 2,2 3 segmentation_increased_gap_chance
submit_job 10 0 0.3 7 1.5 "dice" 32,64,128 2,2 3 segmentation_decreased_num_lines
submit_job 10 0 0.3 30 1.5 "dice" 32,64,128 2,2 3 segmentation_increased_num_lines
submit_job 10 0 0.3 15 1.0 "dice" 32,64,128 2,2 3 segmentation_wobble_10
submit_job 10 0 0.3 15 2.0 "dice" 32,64,128 2,2 3 segmentation_wobble_20
submit_job 10 0 0.3 15 3.0 "dice" 32,64,128 2,2 3 segmentation_wobble_30
