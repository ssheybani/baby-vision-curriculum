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
#SBATCH --job-name=job_g2m15
#SBATCH --output=job_g2m15_Out
#SBATCH --error=job_g2m15_Err

######  Module commands #####
module load deeplearning
ulimit -u 20000


######  Job commands go below this line #####
### Hyperparameters and other arguments
train_group='g1'
jpg_root='/N/project/infant_image_statistics/preproc_saber/JPG_10fps/'
savedir='/N/project/baby_vision_curriculum/trained_models/generative/v2/'


# Initialization
# root: /N/project/baby_vision_curriculum/trained_models/generative/v1/
# models: model_g2_seed_101_other_205_mask15small2.pt
init_checkpoint_path='/N/project/baby_vision_curriculum/trained_models/generative/v1/model_g2_seed_101_other_205_mask15small2.pt'



#keep lr/batch_size at ~1e-4 unless you want to experiment with the stochasticity of SGD.
ds_rate=1
mask_ratio=0.15
lr=0.001
batch_size=16
architecture='small2'
n_epoch=2

data_seed=101
other_seed=120
script='pretrain_videomae_v2_saber.py'
other_id='mask15_small2_pre.g2'



python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/generative/scripts/pretrain_videomae_v2_saber.py -train_group $prot -jpg_root $jpg_root -savedir $savedir --init_checkpoint_path $init_checkpoint_path --ds_rate $ds_rate --mask_ratio $mask_ratio --lr $lr --batch_size $batch_size --architecture $architecture --n_epoch $n_epoch --data_seed $data_seed --other_seed $other_seed  --script $script --other_id $other_id