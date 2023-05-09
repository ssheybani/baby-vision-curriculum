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
#SBATCH --job-name=job_tg0g2
#SBATCH --output=job_tg0g2_Out
#SBATCH --error=job_tg0g2_Err

######  Module commands #####
module load deeplearning
ulimit -u 20000


######  Job commands go below this line #####
### Hyperparameters and other arguments

jpg_root='/N/project/infant_image_statistics/preproc_saber/JPG_30fps/'

#keep lr/batch_size at ~1e-4 unless you want to experiment with the stochasticity of SGD.

#@@@@@ debug. also fix the i_break in the code
n_epoch=30
optim='sgd'
lr=0.1
wd=5e-5
momentum=0.9
#sgd,0.1
#adam,0.001
ds_rate=3
mask_ratio=0.5


batch_size=16
architecture='small2'


train_group='g0'
savedir='/N/project/baby_vision_curriculum/trained_models/generative/v2/'

# Initialization
# root: /N/project/baby_vision_curriculum/trained_models/generative/v1/
# models: model_g2_seed_101_other_205_mask15small2.pt
init_checkpoint_path='na'

data_seed=8002
other_seed=8002
script='pretrain_videomae_v2.1.py'
other_id='sgd_mmnt9'

python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/generative/scripts/pretrain_videomae_v2.1.py -train_group $train_group -jpg_root $jpg_root -savedir $savedir --init_checkpoint_path $init_checkpoint_path --ds_rate $ds_rate --mask_ratio $mask_ratio --lr $lr --batch_size $batch_size --architecture $architecture --n_epoch $n_epoch --data_seed $data_seed --other_seed $other_seed  --script $script --other_id $other_id --optim $optim --wd $wd --momentum $momentum


#-----------------------
# Stage 2
# new fname:
model_fname="model_${train_group}_seed_${data_seed}_other_${other_seed}_${other_id}.pt"
init_checkpoint_path="${savedir}${model_fname}"

echo "init_checkpoint_path: $init_checkpoint_path"

#model_fname = '_'.join(['model', train_group, 'seed',str(seed), 'other', str(other_seed), args.other_id])+'.pt'
#         MODELPATH = os.path.join(model_dir,model_fname)
# check if the fpath exists
# then train a new stage

train_group='g2'
savedir='/N/project/baby_vision_curriculum/trained_models/generative/v2/s2/'


data_seed=8002
other_seed=8002
script='pretrain_videomae_v2.1.py'
other_id='sgd_mmnt9_pre.g0'

python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/generative/scripts/pretrain_videomae_v2.1.py -train_group $train_group -jpg_root $jpg_root -savedir $savedir --init_checkpoint_path $init_checkpoint_path --ds_rate $ds_rate --mask_ratio $mask_ratio --lr $lr --batch_size $batch_size --architecture $architecture --n_epoch $n_epoch --data_seed $data_seed --other_seed $other_seed  --script $script --other_id $other_id --optim $optim --wd $wd --momentum $momentum

#-----------------------
# Stage 3
# new fname:
# model_fname="model_${train_group}_seed_${data_seed}_other_${other_seed}_${other_id}.pt"
# init_checkpoint_path="${savedir}${model_fname}"

# echo "init_checkpoint_path: $init_checkpoint_path"

# #model_fname = '_'.join(['model', train_group, 'seed',str(seed), 'other', str(other_seed), args.other_id])+'.pt'
# #         MODELPATH = os.path.join(model_dir,model_fname)
# # check if the fpath exists
# # then train a new stage

# train_group='g2'
# savedir='/N/project/baby_vision_curriculum/trained_models/generative/v2/s3/'


# data_seed=8000
# other_seed=8000
# script='pretrain_videomae_v2.1.py'
# other_id='mask50_small2_pre.g0g1_10ep'

# python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/generative/scripts/pretrain_videomae_v2.1.py -train_group $train_group -jpg_root $jpg_root -savedir $savedir --init_checkpoint_path $init_checkpoint_path --ds_rate $ds_rate --mask_ratio $mask_ratio --lr $lr --batch_size $batch_size --architecture $architecture --n_epoch $n_epoch --data_seed $data_seed --other_seed $other_seed  --script $script --other_id $other_id --optim $optim --wd $wd --momentum $momentum