#!/bin/bash
#####  Constructed by HPC everywhere #####
#SBATCH --mail-user=sheybani@iu.edu
#SBATCH -A r00117
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=1-05:00:00
#SBATCH --partition=gpu
#SBATCH --gpus 4
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --job-name=job_0_d284
#SBATCH --output=job_0_d284_Out
#SBATCH --error=job_0_d284_Err

######  Module commands #####
module load python/gpu
ulimit -u 20000

######  Job commands go below this line #####
### Hyperparameters and other arguments

# Note for job submitters: For each run, modify the data_seed as well as the job-name and output and error file name (Line 16-18).

folder_id='sep28'
seed=284 # Modify this for different runs: 681-683
# to avoid race condition with the other scripts.
sleep_duration=$((180 * (seed % 3)))
echo sleeping for $sleep_duration seconds
sleep $sleep_duration

# jpg_root='/N/project/infant_image_statistics/preproc_saber/JPG_10fps/'
jpg_root='/N/project/baby_vision_curriculum/homeview_subset_30fps/'
savedir="/N/project/baby_vision_curriculum/trained_models/generative/v3/${folder_id}/"
tbsavedir="/N/project/baby_vision_curriculum/trained_models/generative/v3/${folder_id}/benchmarks/toybox/"
ucsavedir="/N/project/baby_vision_curriculum/trained_models/generative/v3/${folder_id}/benchmarks/ucf101/"
sssavedir="/N/project/baby_vision_curriculum/trained_models/generative/v3/${folder_id}/benchmarks/ssv2/"

# Downstream tasks
ssv2_root='/N/project/baby_vision_curriculum/benchmarks/ssv2/easy_frames/'
toybox_root='/N/project/baby_vision_curriculum/benchmarks/toybox/vids/toybox/'

n_epoch=5
curr='dev'
condition='default'
ds_rate=1
tubelet_size=2
num_frames=$((8*tubelet_size))
tbframe_rate=5
ucframe_rate=10
ssframe_rate=6
batch_size=16
n_trainsamples=128000
max_epoch_iters=2000
tbbatch_size=64
ucbatch_size=64
ssbatch_size=64
mask_sampler='tube'
mask_ratio=0.9

optim='sgd'
lr=0.1
wd=0
architecture='base'
momentum=0.9
keep_val='n'

fold=0
stage=0
train_group='na'
run_id="${curr}_${stage}_${train_group}_${condition}_${fold}_${seed}"
init_checkpoint_path='na'

echo "run id: $run_id"

##----------------------------
# Compute Downstream Embeddings
num_workers=6

ds_task='ssv2'
python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_videomae.py -ds_task $ds_task -vid_root $ssv2_root -init_checkpoint_path $init_checkpoint_path -savedir $sssavedir --frame_rate $ssframe_rate --run_id $run_id --batch_size $ssbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size

ds_task='toybox'
python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_videomae.py -ds_task $ds_task -vid_root $toybox_root -init_checkpoint_path $init_checkpoint_path -savedir $tbsavedir --frame_rate $tbframe_rate --run_id $run_id --batch_size $tbbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size

ds_task='ucf101'
python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_videomae.py -ds_task $ds_task -vid_root $toybox_root -init_checkpoint_path $init_checkpoint_path -savedir $ucsavedir --frame_rate $ucframe_rate --run_id $run_id --batch_size $ucbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size


#________________________________________________________________
# #-----------------------
# # Stage 1

stage=1
train_group='g0'
fold=$(( (seed + stage) % 3 ))
run_id="${curr}_${stage}_${train_group}_${condition}_${fold}_${seed}"


python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/pretraining/generative/scripts/pretrain_videomae_v3.1.py -jpg_root $jpg_root -train_group $train_group -init_checkpoint_path $init_checkpoint_path -savedir $savedir --ds_rate $ds_rate --lr $lr --momentum $momentum --wd $wd --batch_size $batch_size --architecture $architecture --n_epoch $n_epoch --seed $seed --optim $optim --condition $condition --fold $fold --num_frames $num_frames --n_trainsamples $n_trainsamples --max_epoch_iters $max_epoch_iters --tubelet_size $tubelet_size --run_id $run_id --mask_sampler $mask_sampler --mask_ratio $mask_ratio --keep_val $keep_val

model_fname="model_${run_id}.pth.tar"
init_checkpoint_path="${savedir}${model_fname}"
echo "init_checkpoint_path: $init_checkpoint_path"

##----------------------------
# Compute Downstream Embeddings
# num_workers=6

# ds_task='ssv2'
# vid_root=$ssv2_root
# python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_videomae.py -ds_task $ds_task -vid_root $vid_root -init_checkpoint_path $init_checkpoint_path -savedir $sssavedir --frame_rate $ssframe_rate --run_id $run_id --batch_size $ssbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size

# ds_task='toybox'
# python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_videomae.py -ds_task $ds_task -vid_root $toybox_root -init_checkpoint_path $init_checkpoint_path -savedir $tbsavedir --frame_rate $tbframe_rate --run_id $run_id --batch_size $tbbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size

# ds_task='ucf101'
# python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_videomae.py -ds_task $ds_task -vid_root $toybox_root -init_checkpoint_path $init_checkpoint_path -savedir $ucsavedir --frame_rate $ucframe_rate --run_id $run_id --batch_size $ucbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size



#________________________________________________________________
# #-----------------------
# # Stage 2
stage=2
train_group='g1'
fold=$(( (seed + stage) % 3 ))
run_id="${curr}_${stage}_${train_group}_${condition}_${fold}_${seed}"

python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/pretraining/generative/scripts/pretrain_videomae_v3.1.py -jpg_root $jpg_root -train_group $train_group -init_checkpoint_path $init_checkpoint_path -savedir $savedir --ds_rate $ds_rate --lr $lr --momentum $momentum --wd $wd --batch_size $batch_size --architecture $architecture --n_epoch $n_epoch --seed $seed --optim $optim --condition $condition --fold $fold --num_frames $num_frames --n_trainsamples $n_trainsamples --max_epoch_iters $max_epoch_iters --tubelet_size $tubelet_size --run_id $run_id --mask_sampler $mask_sampler --mask_ratio $mask_ratio --keep_val $keep_val

model_fname="model_${run_id}.pth.tar"
init_checkpoint_path="${savedir}${model_fname}"
echo "init_checkpoint_path: $init_checkpoint_path"


##----------------------------
# Compute Downstream Embeddings

# ds_task='ssv2'
# vid_root=$ssv2_root
# python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_videomae.py -ds_task $ds_task -vid_root $vid_root -init_checkpoint_path $init_checkpoint_path -savedir $sssavedir --frame_rate $ssframe_rate --run_id $run_id --batch_size $ssbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size

# ds_task='toybox'
# python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_videomae.py -ds_task $ds_task -vid_root $toybox_root -init_checkpoint_path $init_checkpoint_path -savedir $tbsavedir --frame_rate $tbframe_rate --run_id $run_id --batch_size $tbbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size

# ds_task='ucf101'
# python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_videomae.py -ds_task $ds_task -vid_root $toybox_root -init_checkpoint_path $init_checkpoint_path -savedir $ucsavedir --frame_rate $ucframe_rate --run_id $run_id --batch_size $ucbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size


#________________________________________________________________
# #-----------------------
# # Stage 3
stage=3
train_group='g2'
fold=$(( (seed + stage) % 3 ))
run_id="${curr}_${stage}_${train_group}_${condition}_${fold}_${seed}"

python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/pretraining/generative/scripts/pretrain_videomae_v3.1.py -jpg_root $jpg_root -train_group $train_group -init_checkpoint_path $init_checkpoint_path -savedir $savedir --ds_rate $ds_rate --lr $lr --momentum $momentum --wd $wd --batch_size $batch_size --architecture $architecture --n_epoch $n_epoch --seed $seed --optim $optim --condition $condition --fold $fold --num_frames $num_frames --n_trainsamples $n_trainsamples --max_epoch_iters $max_epoch_iters --tubelet_size $tubelet_size --run_id $run_id --mask_sampler $mask_sampler --mask_ratio $mask_ratio --keep_val $keep_val

model_fname="model_${run_id}.pth.tar"
init_checkpoint_path="${savedir}${model_fname}"
echo "init_checkpoint_path: $init_checkpoint_path"

##----------------------------
# Compute Downstream Embeddings

ds_task='ssv2'
vid_root=$ssv2_root
checkpoint_dir=$savedir
python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_videomae.py -ds_task $ds_task -vid_root $vid_root -init_checkpoint_path $init_checkpoint_path -savedir $sssavedir --frame_rate $ssframe_rate --run_id $run_id --batch_size $ssbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size --checkpoint_dir $checkpoint_dir

ds_task='toybox'
python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_videomae.py -ds_task $ds_task -vid_root $toybox_root -init_checkpoint_path $init_checkpoint_path -savedir $tbsavedir --frame_rate $tbframe_rate --run_id $run_id --batch_size $tbbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size --checkpoint_dir $checkpoint_dir

ds_task='ucf101'
python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_videomae.py -ds_task $ds_task -vid_root $toybox_root -init_checkpoint_path $init_checkpoint_path -savedir $ucsavedir --frame_rate $ucframe_rate --run_id $run_id --batch_size $ucbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size --checkpoint_dir $checkpoint_dir
