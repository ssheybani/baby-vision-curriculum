#!/bin/bash
#####  Constructed by HPC everywhere #####
#SBATCH --mail-user=sheybani@iu.edu
#SBATCH -A r00117
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=0-03:00:00
#SBATCH --partition=gpu
#SBATCH --gpus 4
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --job-name=job_0_d312
#SBATCH --output=job_0_d312_Out
#SBATCH --error=job_0_d312_Err

######  Module commands #####
module load python/gpu
ulimit -u 20000

######  Job commands go below this line #####
### Hyperparameters and other arguments

# Note for job submitters: For each run, modify the data_seed as well as the job-name and output and error file name (Line 16-18).

folder_id='jul312'
seed=312 # Modify this for different runs: 681-683
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

n_epoch=1
curr='dev'
train_group='g0'
condition='def'
ds_rate=1
tubelet_size=2
num_frames=$((8*tubelet_size))
tbframe_rate=5
ssframe_rate=6
batch_size=16
n_trainsamples=162000
max_epoch_iters=200
tbbatch_size=64
ucbatch_size=64
ssbatch_size=64
mask_sampler='tube'
mask_ratio=0.9

architecture='base'
optim='sgd'
lr=0.1
wd=0
architecture='base'
momentum=0.9

fold=$((data_seed % 3))
stage=1
run_id="${curr}_${stage}_${train_group}_${condition}_${fold}_${seed}"
init_checkpoint_path="na"

echo "run id: $run_id"

python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/generative/scripts/pretrain_videomae_v3.py -jpg_root $jpg_root -train_group $train_group -init_checkpoint_path $init_checkpoint_path -savedir $savedir --ds_rate $ds_rate --lr $lr --momentum $momentum --wd $wd --batch_size $batch_size --architecture $architecture --n_epoch $n_epoch --seed $seed --optim $optim --condition $condition --fold $fold --num_frames $num_frames --n_trainsamples $n_trainsamples --max_epoch_iters $max_epoch_iters --tubelet_size $tubelet_size --run_id $run_id --mask_sampler $mask_sampler --mask_ratio $mask_ratio
