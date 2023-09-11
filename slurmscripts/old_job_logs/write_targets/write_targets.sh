#!/bin/bash

#SBATCH --job-name write_targets
#SBATCH --partition=gpu
#SBATCH --gpus 4
#SBATCH -o output.txt
#SBATCH -e error.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hhansar@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu 8G
#SBATCH --time=10:00:00
#SBATCH -A r00117


#Load any modules that your program needs
module load deeplearning/2.9.1

######  Job commands go below this line #####
echo '###### move to script dir ######'
cd /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/imagenet/
echo '###### Directory Changed! ######'

echo '###### Running benchmarking.py ######'
python3 ./make_in_subset_Himanshu.py
echo '###### Run Complete! ######'
