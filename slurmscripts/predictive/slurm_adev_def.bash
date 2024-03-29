#!/bin/bash
#####  Constructed by HPC everywhere #####
#SBATCH --mail-user=sheybani@iu.edu
#SBATCH -A r00117
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=0-10:00:00
#SBATCH --partition=gpu
#SBATCH --gpus 4
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --job-name=job_2_d112
#SBATCH --output=job_2_d112_Out
#SBATCH --error=job_2_d112_Err

######  Module commands #####
module load python/gpu
ulimit -u 20000

######  Job commands go below this line #####
### Hyperparameters and other arguments

# Note for job submitters: For each run, modify the data_seed as well as the job-name and output and error file name (Line 16-18).

folder_id='aug1'
seed=112 # Modify this for different runs: 681-683
# to avoid race condition with the other scripts.
sleep_duration=$((180 * (seed % 3)))
echo sleeping for $sleep_duration seconds
sleep $sleep_duration

jpg_root='/N/project/baby_vision_curriculum/homeview_subset_30fps/'
savedir="/N/project/baby_vision_curriculum/trained_models/predictive/v1/${folder_id}/"
tbsavedir="/N/project/baby_vision_curriculum/trained_models/predictive/v1/${folder_id}/benchmarks/toybox/"
ucsavedir="/N/project/baby_vision_curriculum/trained_models/predictive/v1/${folder_id}/benchmarks/ucf101/"
sssavedir="/N/project/baby_vision_curriculum/trained_models/predictive/v1/${folder_id}/benchmarks/ssv2/"
cfsavedir="/N/project/baby_vision_curriculum/trained_models/predictive/v1/${folder_id}/benchmarks/cifar10/"

# Downstream tasks
ssv2_root='/N/project/baby_vision_curriculum/benchmarks/ssv2/easy_frames/'
toybox_root='/N/project/baby_vision_curriculum/benchmarks/toybox/vids/toybox/'

#@@@@@ debug. also fix the i_break in the code
n_epoch=2
curr='adev'
condition='default'
ds_rate=1
interval=$((10 * (30 / ds_rate)))
tubelet_size=1
num_frames=$((2*tubelet_size))
tbframe_rate=2
# 5
ucframe_rate=3
# 10
ssframe_rate=1
# 6
batch_size=16
n_trainsamples=1280000
max_epoch_iters=5000
tbbatch_size=64
ucbatch_size=64
ssbatch_size=64
cfbatch_size=64
enc_mask_scale=0.85
pred_mask_scale=0.1
augs='cg'

optim='sgd'
lr=0.03
momentum=0.9
wd=0
architecture='base'
allow_overlap='n'

fold=$((seed % 3))

stage=0
train_group='na'
run_id="${curr}_${stage}_${train_group}_${condition}_${fold}_${seed}"
init_checkpoint_path='na'

##----------------------------
# Compute Downstream Embeddings
num_workers=6

ds_task='ssv2'
vid_root=$ssv2_root
python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_jepa.py -ds_task $ds_task -vid_root $vid_root -init_checkpoint_path $init_checkpoint_path -savedir $sssavedir --frame_rate $ssframe_rate --run_id $run_id --batch_size $ssbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size

ds_task='toybox'
python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_jepa.py -ds_task $ds_task -vid_root $toybox_root -init_checkpoint_path $init_checkpoint_path -savedir $tbsavedir --frame_rate $tbframe_rate --run_id $run_id --batch_size $tbbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size

ds_task='ucf101'
python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_jepa.py -ds_task $ds_task -vid_root $toybox_root -init_checkpoint_path $init_checkpoint_path -savedir $ucsavedir --frame_rate $ucframe_rate --run_id $run_id --batch_size $ucbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size

ds_task='cifar10'
python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_jepa.py -ds_task $ds_task -vid_root $toybox_root -init_checkpoint_path $init_checkpoint_path -savedir $cfsavedir --frame_rate $tbframe_rate --run_id $run_id --batch_size $cfbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size


#________________________________________________________________
# #-----------------------
# # Stage 1
stage=1
train_group='g2'
run_id="${curr}_${stage}_${train_group}_${condition}_${fold}_${seed}"

python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/predictive/scripts/pretrain_vjepa_v1.py -train_group $train_group -jpg_root $jpg_root -savedir $savedir --init_checkpoint_path $init_checkpoint_path --ds_rate $ds_rate --lr $lr --wd $wd --batch_size $batch_size --architecture $architecture --n_epoch $n_epoch --seed $seed --run_id $run_id --optim $optim --condition $condition --fold $fold --num_frames $num_frames --n_trainsamples $n_trainsamples --max_epoch_iters $max_epoch_iters --tubelet_size $tubelet_size --pred_mask_scale $pred_mask_scale --allow_overlap $allow_overlap --momentum $momentum --enc_mask_scale $enc_mask_scale --interval $interval --augs $augs

model_fname="model_${run_id}.pth.tar"
init_checkpoint_path="${savedir}${model_fname}"
echo "init_checkpoint_path: $init_checkpoint_path"


##----------------------------
# Compute Downstream Embeddings
num_workers=6

ds_task='ssv2'
vid_root=$ssv2_root
python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_jepa.py -ds_task $ds_task -vid_root $vid_root -init_checkpoint_path $init_checkpoint_path -savedir $sssavedir --frame_rate $ssframe_rate --run_id $run_id --batch_size $ssbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size

ds_task='toybox'
python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_jepa.py -ds_task $ds_task -vid_root $toybox_root -init_checkpoint_path $init_checkpoint_path -savedir $tbsavedir --frame_rate $tbframe_rate --run_id $run_id --batch_size $tbbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size

ds_task='ucf101'
python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_jepa.py -ds_task $ds_task -vid_root $toybox_root -init_checkpoint_path $init_checkpoint_path -savedir $ucsavedir --frame_rate $ucframe_rate --run_id $run_id --batch_size $ucbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size

ds_task='cifar10'
python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_jepa.py -ds_task $ds_task -vid_root $toybox_root -init_checkpoint_path $init_checkpoint_path -savedir $cfsavedir --frame_rate $tbframe_rate --run_id $run_id --batch_size $cfbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size



#________________________________________________________________
# #-----------------------
# # Stage 2
stage=2
train_group='g1'
run_id="${curr}_${stage}_${train_group}_${condition}_${fold}_${seed}"

python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/predictive/scripts/pretrain_vjepa_v1.py -train_group $train_group -jpg_root $jpg_root -savedir $savedir --init_checkpoint_path $init_checkpoint_path --ds_rate $ds_rate --lr $lr --wd $wd --batch_size $batch_size --architecture $architecture --n_epoch $n_epoch --seed $seed --run_id $run_id --optim $optim --condition $condition --fold $fold --num_frames $num_frames --n_trainsamples $n_trainsamples --max_epoch_iters $max_epoch_iters --tubelet_size $tubelet_size --pred_mask_scale $pred_mask_scale --allow_overlap $allow_overlap --momentum $momentum --enc_mask_scale $enc_mask_scale --interval $interval --augs $augs

##----------------------------
# Compute Downstream Embeddings

ds_task='ssv2'
vid_root=$ssv2_root
python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_jepa.py -ds_task $ds_task -vid_root $vid_root -init_checkpoint_path $init_checkpoint_path -savedir $sssavedir --frame_rate $ssframe_rate --run_id $run_id --batch_size $ssbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size

ds_task='toybox'
python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_jepa.py -ds_task $ds_task -vid_root $toybox_root -init_checkpoint_path $init_checkpoint_path -savedir $tbsavedir --frame_rate $tbframe_rate --run_id $run_id --batch_size $tbbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size

ds_task='ucf101'
python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_jepa.py -ds_task $ds_task -vid_root $toybox_root -init_checkpoint_path $init_checkpoint_path -savedir $ucsavedir --frame_rate $ucframe_rate --run_id $run_id --batch_size $ucbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size

ds_task='cifar10'
python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_jepa.py -ds_task $ds_task -vid_root $toybox_root -init_checkpoint_path $init_checkpoint_path -savedir $cfsavedir --frame_rate $tbframe_rate --run_id $run_id --batch_size $cfbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size


#________________________________________________________________
# #-----------------------
# # Stage 3
stage=3
train_group='g0'
run_id="${curr}_${stage}_${train_group}_${condition}_${fold}_${seed}"

python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/predictive/scripts/pretrain_vjepa_v1.py -train_group $train_group -jpg_root $jpg_root -savedir $savedir --init_checkpoint_path $init_checkpoint_path --ds_rate $ds_rate --lr $lr --wd $wd --batch_size $batch_size --architecture $architecture --n_epoch $n_epoch --seed $seed --run_id $run_id --optim $optim --condition $condition --fold $fold --num_frames $num_frames --n_trainsamples $n_trainsamples --max_epoch_iters $max_epoch_iters --tubelet_size $tubelet_size --pred_mask_scale $pred_mask_scale --allow_overlap $allow_overlap --momentum $momentum --enc_mask_scale $enc_mask_scale --interval $interval --augs $augs

##----------------------------
# Compute Downstream Embeddings

ds_task='ssv2'
vid_root=$ssv2_root
python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_jepa.py -ds_task $ds_task -vid_root $vid_root -init_checkpoint_path $init_checkpoint_path -savedir $sssavedir --frame_rate $ssframe_rate --run_id $run_id --batch_size $ssbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size

ds_task='toybox'
python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_jepa.py -ds_task $ds_task -vid_root $toybox_root -init_checkpoint_path $init_checkpoint_path -savedir $tbsavedir --frame_rate $tbframe_rate --run_id $run_id --batch_size $tbbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size

ds_task='ucf101'
python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_jepa.py -ds_task $ds_task -vid_root $toybox_root -init_checkpoint_path $init_checkpoint_path -savedir $ucsavedir --frame_rate $ucframe_rate --run_id $run_id --batch_size $ucbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size

ds_task='cifar10'
python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_jepa.py -ds_task $ds_task -vid_root $toybox_root -init_checkpoint_path $init_checkpoint_path -savedir $cfsavedir --frame_rate $tbframe_rate --run_id $run_id --batch_size $cfbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size
