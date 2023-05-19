#!/bin/bash
#####  Constructed by HPC everywhere #####
#SBATCH --mail-user=sheybani@iu.edu
#SBATCH -A r00117
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=50
#SBATCH --time=0-05:00:00
#SBATCH --partition=gpu
#SBATCH --gpus 4
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --job-name=job_uuemb
#SBATCH --output=job_uuemb_Out
#SBATCH --error=job_uuemb_Err

######  Module commands #####
module load deeplearning
ulimit -u 20000

######  Job commands go below this line #####
### Hyperparameters and other arguments

# Note for job submitters: For each run, modify the data_seed as well as the job-name and output and error file name (Line 16-18).



#just get the savedir right


# savedir='/N/project/baby_vision_curriculum/trained_models/generative/v2/may14/s2/'
ucsavedir='/N/project/baby_vision_curriculum/trained_models/generative/v2/benchmarks/ucf101/may14/s1/'


architecture='base'
num_workers=6
seed=100
uctask='ucf101'

ucframe_rate=10
ucbatch_size=64
other_id='x'
prot_name='untrained'

# Initialization
init_checkpoint_path='na'

python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_ucf101_embeddings.py -task $uctask -init_checkpoint_path $init_checkpoint_path -savedir $ucsavedir --frame_rate $ucframe_rate --prot_name $prot_name --batch_size $ucbatch_size --architecture $architecture --seed $seed --other_id $other_id --num_workers $num_workers 







