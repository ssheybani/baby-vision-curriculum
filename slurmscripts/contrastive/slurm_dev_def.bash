#!/bin/bash
#####  Constructed by HPC everywhere #####
#SBATCH --mail-user=sheybani@iu.edu
#SBATCH -A r00117
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=0-3:00:00
#SBATCH --partition=gpu
#SBATCH --gpus 4
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --job-name=job_0_d176
#SBATCH --output=job_0_d176_Out
#SBATCH --error=job_0_d176_Err

######  Module commands #####
module load python/gpu
ulimit -u 20000

######  Job commands go below this line #####
### Hyperparameters and other arguments

# Note for job submitters: For each run, modify the data_seed as well as the job-name and output and error file name (Line 16-18).

folder_id='aug73'
seed=176 # Modify this for different runs: 681-683
# to avoid race condition with the other scripts.
sleep_duration=$((180 * (seed % 3)))
echo sleeping for $sleep_duration seconds
sleep $sleep_duration

# jpg_root='/N/project/infant_image_statistics/preproc_saber/JPG_10fps/'
jpg_root='/N/project/baby_vision_curriculum/homeview_subset_30fps/'
savedir="/N/project/baby_vision_curriculum/trained_models/contrastive/v1/${folder_id}/"
tbsavedir="/N/project/baby_vision_curriculum/trained_models/contrastive/v1/${folder_id}/benchmarks/toybox/"
ucsavedir="/N/project/baby_vision_curriculum/trained_models/contrastive/v1/${folder_id}/benchmarks/ucf101/"
sssavedir="/N/project/baby_vision_curriculum/trained_models/contrastive/v1/${folder_id}/benchmarks/ssv2/"
cfsavedir="/N/project/baby_vision_curriculum/trained_models/contrastive/v1/${folder_id}/benchmarks/cifar10/"

# Downstream tasks
ssv2_root='/N/project/baby_vision_curriculum/benchmarks/ssv2/easy_frames/'
toybox_root='/N/project/baby_vision_curriculum/benchmarks/toybox/vids/toybox/'

#@@@@@ debug. also fix the i_break in the code
n_epoch=1
curr='dev'
condition='default'
ds_rate=1
tubelet_size=1
num_frames=2
tbframe_rate=2
# 5
ucframe_rate=3
# 10
ssframe_rate=1
# 6
batch_size=32
max_epoch_iters=1500
n_trainsamples=192000
# 4*batch_size*max_epoch_iters
# 1280000
tbbatch_size=64
ucbatch_size=64
ssbatch_size=64
cfbatch_size=64
# enc_mask_scale=0.85
augs='cjo'
# 'cg'
pred_emb_dim=512
# 1024
# 2048

optim='sgd'
momentum=0.9
wd=0
architecture='resnet18'

num_workers=6

fold=$((seed % 3))

stage=0
train_group='na'
run_id="${curr}_${stage}_${train_group}_${condition}_${fold}_${seed}"
init_checkpoint_path='na'

##----------------------------
# Compute Downstream Embeddings
num_frames=2
ds_task='cifar10'
python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_simclr.py -ds_task $ds_task -vid_root $toybox_root -init_checkpoint_path $init_checkpoint_path -savedir $cfsavedir --frame_rate $tbframe_rate --run_id $run_id --batch_size $cfbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size --pred_emb_dim $pred_emb_dim

ds_task='toybox'
python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_simclr.py -ds_task $ds_task -vid_root $toybox_root -init_checkpoint_path $init_checkpoint_path -savedir $tbsavedir --frame_rate $tbframe_rate --run_id $run_id --batch_size $tbbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size --pred_emb_dim $pred_emb_dim



#________________________________________________________________
# #-----------------------
# # Stage 1
stage=1
train_group='g0'
lr=0.01
interval=$((30 * (30 / ds_rate)))
run_id="${curr}_${stage}_${train_group}_${condition}_${fold}_${seed}"

python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/contrastive/scripts/pretrain_simclr_v1.py -train_group $train_group -jpg_root $jpg_root -savedir $savedir --init_checkpoint_path $init_checkpoint_path --ds_rate $ds_rate --lr $lr --wd $wd --batch_size $batch_size --architecture $architecture --n_epoch $n_epoch --seed $seed --run_id $run_id --optim $optim --condition $condition --fold $fold --n_trainsamples $n_trainsamples --max_epoch_iters $max_epoch_iters --momentum $momentum --interval $interval --augs $augs --pred_emb_dim $pred_emb_dim

model_fname="model_${run_id}.pth.tar"
init_checkpoint_path="${savedir}${model_fname}"
echo "init_checkpoint_path: $init_checkpoint_path"


##----------------------------
# Compute Downstream Embeddings

ds_task='cifar10'
python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_simclr.py -ds_task $ds_task -vid_root $toybox_root -init_checkpoint_path $init_checkpoint_path -savedir $cfsavedir --frame_rate $tbframe_rate --run_id $run_id --batch_size $cfbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size --pred_emb_dim $pred_emb_dim

ds_task='toybox'
python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_simclr.py -ds_task $ds_task -vid_root $toybox_root -init_checkpoint_path $init_checkpoint_path -savedir $tbsavedir --frame_rate $tbframe_rate --run_id $run_id --batch_size $tbbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size --pred_emb_dim $pred_emb_dim




#________________________________________________________________
# #-----------------------
# # Stage 2
stage=2
train_group='g1'
lr=0.001
interval=$((3 * (30 / ds_rate)))
run_id="${curr}_${stage}_${train_group}_${condition}_${fold}_${seed}"

python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/contrastive/scripts/pretrain_simclr_v1.py -train_group $train_group -jpg_root $jpg_root -savedir $savedir --init_checkpoint_path $init_checkpoint_path --ds_rate $ds_rate --lr $lr --wd $wd --batch_size $batch_size --architecture $architecture --n_epoch $n_epoch --seed $seed --run_id $run_id --optim $optim --condition $condition --fold $fold --n_trainsamples $n_trainsamples --max_epoch_iters $max_epoch_iters --momentum $momentum --interval $interval --augs $augs --pred_emb_dim $pred_emb_dim

model_fname="model_${run_id}.pth.tar"
init_checkpoint_path="${savedir}${model_fname}"
echo "init_checkpoint_path: $init_checkpoint_path"


##----------------------------
# Compute Downstream Embeddings

ds_task='cifar10'
python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_simclr.py -ds_task $ds_task -vid_root $toybox_root -init_checkpoint_path $init_checkpoint_path -savedir $cfsavedir --frame_rate $tbframe_rate --run_id $run_id --batch_size $cfbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size --pred_emb_dim $pred_emb_dim

ds_task='toybox'
python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_simclr.py -ds_task $ds_task -vid_root $toybox_root -init_checkpoint_path $init_checkpoint_path -savedir $tbsavedir --frame_rate $tbframe_rate --run_id $run_id --batch_size $tbbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size --pred_emb_dim $pred_emb_dim

#________________________________________________________________
# #-----------------------
# # Stage 3
stage=3
lr=0.0001
interval=$((3 * (3 / ds_rate)))
train_group='g2'
run_id="${curr}_${stage}_${train_group}_${condition}_${fold}_${seed}"

python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/contrastive/scripts/pretrain_simclr_v1.py -train_group $train_group -jpg_root $jpg_root -savedir $savedir --init_checkpoint_path $init_checkpoint_path --ds_rate $ds_rate --lr $lr --wd $wd --batch_size $batch_size --architecture $architecture --n_epoch $n_epoch --seed $seed --run_id $run_id --optim $optim --condition $condition --fold $fold --n_trainsamples $n_trainsamples --max_epoch_iters $max_epoch_iters --momentum $momentum --interval $interval --augs $augs --pred_emb_dim $pred_emb_dim

model_fname="model_${run_id}.pth.tar"
init_checkpoint_path="${savedir}${model_fname}"
echo "init_checkpoint_path: $init_checkpoint_path"


##----------------------------
# Compute Downstream Embeddings

ds_task='cifar10'
python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_simclr.py -ds_task $ds_task -vid_root $toybox_root -init_checkpoint_path $init_checkpoint_path -savedir $cfsavedir --frame_rate $tbframe_rate --run_id $run_id --batch_size $cfbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size --pred_emb_dim $pred_emb_dim

ds_task='toybox'
python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_simclr.py -ds_task $ds_task -vid_root $toybox_root -init_checkpoint_path $init_checkpoint_path -savedir $tbsavedir --frame_rate $tbframe_rate --run_id $run_id --batch_size $tbbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size --pred_emb_dim $pred_emb_dim
