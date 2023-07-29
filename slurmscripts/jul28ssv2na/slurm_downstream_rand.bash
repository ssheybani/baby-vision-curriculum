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
#SBATCH --job-name=job_0_d245
#SBATCH --output=job_0_d245_Out
#SBATCH --error=job_0_d245_Err

######  Module commands #####
module load python/gpu
ulimit -u 20000

folder_id='jul28dev'
seed=142 # Modify this for different runs: 681-683
# to avoid race condition with the other scripts.
sleep_duration=$((180 * (seed % 3)))
echo sleeping for $sleep_duration seconds
sleep $sleep_duration

# jpg_root='/N/project/baby_vision_curriculum/homeview_subset_30fps/'
# savedir="/N/project/baby_vision_curriculum/trained_models/predictive/v1/${folder_id}/"
# tbsavedir="/N/project/baby_vision_curriculum/trained_models/generative/v1/${folder_id}/benchmarks/toybox/"
# ucsavedir="/N/project/baby_vision_curriculum/trained_models/generative/v1/${folder_id}/benchmarks/ucf101/"
sssavedir="/N/project/baby_vision_curriculum/trained_models/generative/v3/${folder_id}/benchmarks/ssv2/"

n_epoch=1
curr='dev'
train_group='g2'
condition='fr6'
# ds_rate=1
# interval=$((30 * (30 / ds_rate)))
tubelet_size=2
num_frames=16
#$((2*tubelet_size))
tbframe_rate=5
ssframe_rate=6
# batch_size=64
# n_trainsamples=1280000
# max_epoch_iters=5000
tbbatch_size=64
ucbatch_size=64
ssbatch_size=64
# enc_mask_scale=0.85
# pred_mask_scale=0.1
# augs='c'
architecture='base'

fold=0
stage=1
run_id="${curr}_${stage}_${train_group}_${condition}_${fold}_${seed}"

savedir='/N/project/baby_vision_curriculum/trained_models/generative/v2/jun11/s3/'
model_fname='model_g2_seed_142_other_142_dev_pre.g0g1.pt'
init_checkpoint_path="${savedir}${model_fname}"
# init_checkpoint_path="na"

echo "init_checkpoint_path: $init_checkpoint_path"

##----------------------------
# Compute SSv2 Embeddings
ssv2_root='/N/project/baby_vision_curriculum/benchmarks/ssv2/easy_frames/'

num_workers=6

python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_ssv2_embeddings.py -vid_root $ssv2_root -init_checkpoint_path $init_checkpoint_path -savedir $sssavedir --frame_rate $ssframe_rate --run_id $run_id --batch_size $ssbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size

