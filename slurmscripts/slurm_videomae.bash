#!/bin/bash
#####  Constructed by HPC everywhere #####
#SBATCH --mail-user=sheybani@iu.edu
#SBATCH -A r00117
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=50
#SBATCH --time=0-04:00:00
#SBATCH --partition=gpu
#SBATCH --gpus 4
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --job-name=job_g2g1g0
#SBATCH --output=job_g2g1g0_Out
#SBATCH --error=job_g2g1g0_Err

######  Module commands #####
module load deeplearning
ulimit -u 20000


######  Job commands go below this line #####
### Hyperparameters and other arguments

script='pretrain_videomae_v2.py'
jpg_root='/N/project/infant_image_statistics/preproc_saber/JPG_10fps/'

#keep lr/batch_size at ~1e-4 unless you want to experiment with the stochasticity of SGD.

#@@@@@ debug. also fix the i_break in the code
n_epoch=10

ds_rate=1
mask_ratio=0.5
lr=0.001
batch_size=16
architecture='small2'


# Initialization
#-----------------------
chpt_dir='/N/project/baby_vision_curriculum/trained_models/generative/v2/s2/'
chpt_fname="model_g1_seed_1133_other_1345_mask50_small2_pre.g2.pt"
init_checkpoint_path="${chpt_dir}${chpt_fname}"
#'na'
echo "init_checkpoint_path: $init_checkpoint_path"

train_group='g0'
savedir='/N/project/baby_vision_curriculum/trained_models/generative/v2/s3/'


data_seed=2323
other_seed=2233
other_id='mask50_small2_pre.g2g1'

python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/generative/scripts/pretrain_videomae_v2.py -train_group $train_group -jpg_root $jpg_root -savedir $savedir --init_checkpoint_path $init_checkpoint_path --ds_rate $ds_rate --mask_ratio $mask_ratio --lr $lr --batch_size $batch_size --architecture $architecture --n_epoch $n_epoch --data_seed $data_seed --other_seed $other_seed  --script $script --other_id $other_id
