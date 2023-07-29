#!/bin/bash
#####  Constructed by HPC everywhere #####
#SBATCH --mail-user=sheybani@iu.edu
#SBATCH -A r00117
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=0-06:00:00
#SBATCH --partition=gpu
#SBATCH --gpus 4
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --job-name=job_210_212
#SBATCH --output=job_210_212_Out
#SBATCH --error=job_210_212_Err

######  Module commands #####
module load python/gpu
ulimit -u 20000

######  Job commands go below this line #####
### Hyperparameters and other arguments

# Note for job submitters: For each run, modify the data_seed as well as the job-name and output and error file name (Line 16-18).

data_seed=212 # Modify this for different runs: 681-683
# to avoid race condition with the other scripts.
sleep_duration=$((180 * (data_seed % 3)))
echo sleeping for $sleep_duration seconds
sleep $sleep_duration

# jpg_root='/N/project/infant_image_statistics/preproc_saber/JPG_10fps/'
jpg_root='/N/project/infant_image_statistics/preproc_saber/JPG_30fps/'

saveroot='/N/project/baby_vision_curriculum/trained_models/generative/v2/jun21/'
tbsaveroot='/N/project/baby_vision_curriculum/trained_models/generative/v2/benchmarks/toybox/jun21/'
ucsaveroot='/N/project/baby_vision_curriculum/trained_models/generative/v2/benchmarks/ucf101/jun21/'

#@@@@@ debug. also fix the i_break in the code
n_epoch=5
curr='adev'
condition='default'
other_id="${curr}_${condition}"
monitor='grad'
ds_rate=1
batch_size=16
n_trainsamples=162000
max_epoch_iters=2000
tbbatch_size=64
ucbatch_size=64
mask_sampler='tube'
tubelet_size=1
num_frames=1

train_group='g2'
fold=$((data_seed % 3))
mask_ratio=0.9
#adamw: lr=1.5e-4, wd=0.05, momentum=0.9 (doesn't get used)
#adam: lr=0.001. wd=1e-4, momentum=0.9 (doesn't get used)
#sgd: 0.1, 0, 0.9
optim='sgd'
lr=0.1
wd=0
architecture='base'
momentum=0.9



savedir="${saveroot}s1/"
# Initialization
init_checkpoint_path='na'

other_seed=$data_seed
script='pretrain_videomae_2.5.py'


python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/generative/scripts/pretrain_videomae_v2.5.py -train_group $train_group -jpg_root $jpg_root -savedir $savedir --init_checkpoint_path $init_checkpoint_path --ds_rate $ds_rate --mask_ratio $mask_ratio --lr $lr --wd $wd --batch_size $batch_size --architecture $architecture --n_epoch $n_epoch --data_seed $data_seed --other_seed $other_seed  --script $script --other_id $other_id --optim $optim  --momentum $momentum --condition $condition --monitor $monitor --fold $fold --num_frames $num_frames --n_trainsamples $n_trainsamples --mask_sampler $mask_sampler --max_epoch_iters $max_epoch_iters --tubelet_size $tubelet_size

model_fname="model_${train_group}_seed_${data_seed}_other_${other_seed}_${other_id}.pt"
init_checkpoint_path="${savedir}${model_fname}"

echo "init_checkpoint_path: $init_checkpoint_path"

##----------------------------
# Compute ToyBox Embeddings
toybox_root='/N/project/baby_vision_curriculum/benchmarks/toybox/vids/toybox/'
tbsavedir="${tbsaveroot}s1/"

prot_name=$train_group
# architecture='base'
# other_id='sgd'
seed=$data_seed
tbframe_rate=5


num_workers=6

python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_toybox_embeddings.py -vid_root $toybox_root -init_checkpoint_path $init_checkpoint_path -savedir $tbsavedir --frame_rate $tbframe_rate --prot_name $prot_name --batch_size $tbbatch_size --architecture $architecture --seed $seed --other_id $other_id --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size


##----------------------------
# Compute UCF101 Embeddings
uctask='ucf101'
ucsavedir="${ucsaveroot}s1/"
ucframe_rate=10

# prot_name=$train_group
# seed=$data_seed
# num_workers=6

python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_ucf101_embeddings.py -task $uctask -init_checkpoint_path $init_checkpoint_path -savedir $ucsavedir --frame_rate $ucframe_rate --prot_name $prot_name --batch_size $ucbatch_size --architecture $architecture --seed $seed --other_id $other_id --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size 


#________________________________________________________________
# #-----------------------
# # Stage 2
# # new fname:
model_fname="model_${train_group}_seed_${data_seed}_other_${other_seed}_${other_id}.pt"
init_checkpoint_path="${savedir}${model_fname}"

echo "init_checkpoint_path: $init_checkpoint_path"
other_id="${other_id}_pre.${train_group}"

train_group='g1'
fold=$(((data_seed + 1) % 3))
savedir="${saveroot}s2/"

python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/generative/scripts/pretrain_videomae_v2.5.py -train_group $train_group -jpg_root $jpg_root -savedir $savedir --init_checkpoint_path $init_checkpoint_path --ds_rate $ds_rate --mask_ratio $mask_ratio --lr $lr --wd $wd --batch_size $batch_size --architecture $architecture --n_epoch $n_epoch --data_seed $data_seed --other_seed $other_seed  --script $script --other_id $other_id --optim $optim  --momentum $momentum --condition $condition --monitor $monitor --fold $fold --num_frames $num_frames --n_trainsamples $n_trainsamples --mask_sampler $mask_sampler --max_epoch_iters $max_epoch_iters --tubelet_size $tubelet_size

model_fname="model_${train_group}_seed_${data_seed}_other_${other_seed}_${other_id}.pt"
init_checkpoint_path="${savedir}${model_fname}"

echo "init_checkpoint_path: $init_checkpoint_path"

##----------------------------
# Compute ToyBox Embeddings
tbsavedir="${tbsaveroot}s2/"

prot_name=$train_group
# architecture='base'
# other_id='sgd'
seed=$data_seed

python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_toybox_embeddings.py -vid_root $toybox_root -init_checkpoint_path $init_checkpoint_path -savedir $tbsavedir --frame_rate $tbframe_rate --prot_name $prot_name --batch_size $tbbatch_size --architecture $architecture --seed $seed --other_id $other_id --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size

##----------------------------
# Compute UCF101 Embeddings
ucsavedir="${ucsaveroot}s2/"
# prot_name=$train_group
# seed=$data_seed
# num_workers=6

python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_ucf101_embeddings.py -task $uctask -init_checkpoint_path $init_checkpoint_path -savedir $ucsavedir --frame_rate $ucframe_rate --prot_name $prot_name --batch_size $ucbatch_size --architecture $architecture --seed $seed --other_id $other_id --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size


# #-----------------------
# # Stage 3
# # new fname:

other_id="${other_id}${train_group}"

train_group='g0'
# other_id="${condition}_pre.g0g1"
fold=$(((data_seed + 2) % 3))
savedir="${saveroot}s3/"

python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/generative/scripts/pretrain_videomae_v2.5.py -train_group $train_group -jpg_root $jpg_root -savedir $savedir --init_checkpoint_path $init_checkpoint_path --ds_rate $ds_rate --mask_ratio $mask_ratio --lr $lr --wd $wd --batch_size $batch_size --architecture $architecture --n_epoch $n_epoch --data_seed $data_seed --other_seed $other_seed  --script $script --other_id $other_id --optim $optim  --momentum $momentum --condition $condition --monitor $monitor --fold $fold --num_frames $num_frames --n_trainsamples $n_trainsamples --mask_sampler $mask_sampler --max_epoch_iters $max_epoch_iters --tubelet_size $tubelet_size


model_fname="model_${train_group}_seed_${data_seed}_other_${other_seed}_${other_id}.pt"
init_checkpoint_path="${savedir}${model_fname}"

echo "init_checkpoint_path: $init_checkpoint_path"

##----------------------------
# Compute ToyBox Embeddings
tbsavedir="${tbsaveroot}s3/"

prot_name=$train_group
# architecture='base'
# other_id='sgd'
seed=$data_seed

python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_toybox_embeddings.py -vid_root $toybox_root -init_checkpoint_path $init_checkpoint_path -savedir $tbsavedir --frame_rate $tbframe_rate --prot_name $prot_name --batch_size $tbbatch_size --architecture $architecture --seed $seed --other_id $other_id --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size


##----------------------------
# Compute UCF101 Embeddings
ucsavedir="${ucsaveroot}s3/"
# prot_name=$train_group
# seed=$data_seed
# num_workers=6

python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_ucf101_embeddings.py -task $uctask -init_checkpoint_path $init_checkpoint_path -savedir $ucsavedir --frame_rate $ucframe_rate --prot_name $prot_name --batch_size $ucbatch_size --architecture $architecture --seed $seed --other_id $other_id --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size
