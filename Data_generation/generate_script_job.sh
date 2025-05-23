#!/bin/bash
#BSUB -J gen_test        # Set job name
#BSUB -q hpc                 # Specify queue (check if "hpc" is correct)
#BSUB -W 10:00                # Set wall-clock time (2 hours)
#BSUB -n 24
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"   # Request 4GB of memory
#BSUB -o ./logs/outputLogs/output_gen_%J.log            # Output file
#BSUB -e ./logs/errorLogs/error_gen_%J.log             # Error file

lscpu

source /zhome/1a/a/156609/project/path/.venv/bin/activate

notify_done() {
    msg="$1"
    bsub -q hpc -u s204427@dtu.dk -N -J notify_done \
         -o /dev/null -e /dev/null \
         /bin/bash -c "echo 'DONE: $msg at \$(date)'"
}

# (1) size, (2) num jobs (cores and number to be generated), (3) gap_size, (4) skeleton(bool: 0,1), (5) gap_chance, (6) num_lines, (7) wobble

echo "Running baseline: size 512, cores 24, threshold 10, skeleton 1, gap_size 10, gap_chance 0.3, num_lines 15, wobble 1.5"
python -m Data_generation.generate_test 512 24 10 0 0.3 15 1.5
notify_done "Default: size 512, cores 24, threshold 10, skeleton 1, gap_size 10, gap_chance 0.3, num_lines 15, wobble 1.5"

# Varying gap_size
echo "Varying gap_size to 5..."
python -m Data_generation.generate_test 512 24 5 0 0.3 15 1.5
notify_done "Varying gap_size to 5"

echo "Varying gap_size to 20..."
python -m Data_generation.generate_test 512 24 20 0 0.3 15 1.5
notify_done "Varying gap_size to 20"

# Varying gap_chance
echo "Varying gap_chance to 0.1..."
python -m Data_generation.generate_test 512 24 10 0 0.1 15 1.5
notify_done "Varying gap_chance to 0.1"

echo "Varying gap_chance to 0.5..."
python -m Data_generation.generate_test 512 24 10 0 0.5 15 1.5
notify_done "Varying gap_chance to 0.5"


# Varying num_lines
echo "Varying num_lines to 7..."
python -m Data_generation.generate_test 512 24 10 0 0.3 7 1.5
notify_done "Varying num_lines to 7"

echo "Varying num_lines to 30..."
python -m Data_generation.generate_test 512 24 10 0 0.3 30 1.5
notify_done "Varying num_lines to 30"

# Varying wobble
echo "Varying wobble to 1.0..."
python -m Data_generation.generate_test 512 24 10 0 0.3 15 1.0
notify_done "Varying wobble to 1.0"

echo "Varying wobble to 2.0..."
python -m Data_generation.generate_test 512 24 10 0 0.3 15 2.0
notify_done "Varying wobble to 2.0"

echo "Varying wobble to 3.0..."
python -m Data_generation.generate_test 512 24 10 0 0.3 15 3.0
notify_done "Varying wobble to 3.0"



# echo "Size: 512, cores 24, threshold 10, seg"
# python -m Data_generation.generate_test 512 24 10 0

# echo "Size: 512, cores 24, threshold 20, seg"
# python -m Data_generation.generate_test 512 24 20 0

# These are good for debugging
# echo "Size: 256, cores 24, threshold 10, skel"
# python -m Data_generation.generate_test 256 24 10 1

# echo "Size: 256, cores 20, threshold 10, seg"
# python -m Data_generation.generate_test 256 24 10 0


