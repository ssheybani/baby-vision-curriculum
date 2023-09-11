#!/bin/bash
#####  Constructed by HPC everywhere #####
#SBATCH --mail-user=sheybani@iu.edu
#SBATCH -A r00117
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=50
#SBATCH --time=0-15:00:00
#SBATCH --partition=gpu
#SBATCH --gpus 4
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --job-name=job_210_211
#SBATCH --output=job_210_211_Out
#SBATCH --error=job_210_211_Err

######  Module commands #####
module load deeplearning
ulimit -u 20000

######  Job commands go below this line #####
### Hyperparameters and other arguments

# Note for job submitters: For each run, modify the data_seed as well as the job-name and output and error file name (Line 16-18).

data_seed=211 # Modify this for different runs: 211, 212, 213


# jpg_root='/N/project/infant_image_statistics/preproc_saber/JPG_10fps/'
jpg_root='/N/project/infant_image_statistics/preproc_saber/JPG_30fps/'

#@@@@@ debug. also fix the i_break in the code
n_epoch=5
other_id='sgd'

train_group='g2'
ds_rate=1
mask_ratio=0.9
#adamw: lr=1.5e-4, wd=0.05, momentum=0.9 (doesn't get used)
#adam: lr=0.001. wd=1e-4, momentum=0.9 (doesn't get used)
#sgd: 0.1, 0, 0.9
optim='sgd'
lr=0.1
wd=0
batch_size=16
architecture='base'
momentum=0.9

savedir='/N/project/baby_vision_curriculum/trained_models/generative/v2/may14/s1/'

# Initialization
init_checkpoint_path='na'

other_seed=$data_seed
script='pretrain_videomae_2.2.py'


python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/generative/scripts/pretrain_videomae_v2.2.py -train_group $train_group -jpg_root $jpg_root -savedir $savedir --init_checkpoint_path $init_checkpoint_path --ds_rate $ds_rate --mask_ratio $mask_ratio --lr $lr --wd $wd --batch_size $batch_size --architecture $architecture --n_epoch $n_epoch --data_seed $data_seed --other_seed $other_seed  --script $script --other_id $other_id --optim $optim  --momentum $momentum


# #-----------------------
# # Stage 2
# # new fname:
model_fname="model_${train_group}_seed_${data_seed}_other_${other_seed}_${other_id}.pt"
init_checkpoint_path="${savedir}${model_fname}"

echo "init_checkpoint_path: $init_checkpoint_path"

train_group='g1'
other_id='sgd_pre.g2'

savedir='/N/project/baby_vision_curriculum/trained_models/generative/v2/may14/s2/'

python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/generative/scripts/pretrain_videomae_v2.2.py -train_group $train_group -jpg_root $jpg_root -savedir $savedir --init_checkpoint_path $init_checkpoint_path --ds_rate $ds_rate --mask_ratio $mask_ratio --lr $lr --wd $wd --batch_size $batch_size --architecture $architecture --n_epoch $n_epoch --data_seed $data_seed --other_seed $other_seed  --script $script --other_id $other_id --optim $optim  --momentum $momentum

# #-----------------------
# # Stage 3
# # new fname:
model_fname="model_${train_group}_seed_${data_seed}_other_${other_seed}_${other_id}.pt"
init_checkpoint_path="${savedir}${model_fname}"

echo "init_checkpoint_path: $init_checkpoint_path"

train_group='g0'
other_id='sgd_pre.g2g1'

savedir='/N/project/baby_vision_curriculum/trained_models/generative/v2/may14/s3/'

python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/generative/scripts/pretrain_videomae_v2.2.py -train_group $train_group -jpg_root $jpg_root -savedir $savedir --init_checkpoint_path $init_checkpoint_path --ds_rate $ds_rate --mask_ratio $mask_ratio --lr $lr --wd $wd --batch_size $batch_size --architecture $architecture --n_epoch $n_epoch --data_seed $data_seed --other_seed $other_seed  --script $script --other_id $other_id --optim $optim  --momentum $momentum
