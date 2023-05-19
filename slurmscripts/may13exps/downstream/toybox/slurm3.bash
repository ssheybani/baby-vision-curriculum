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
#SBATCH --job-name=job_tadam
#SBATCH --output=job_tadam_Out
#SBATCH --error=job_tadam_Err

######  Module commands #####
module load deeplearning
ulimit -u 20000


vid_root='/N/project/baby_vision_curriculum/benchmarks/toybox/vids/toybox/'
savedir='/N/project/baby_vision_curriculum/trained_models/generative/v2/benchmarks/toybox/'

ch_dir='/N/project/baby_vision_curriculum/trained_models/generative/v2/may13/'

# train duration and lr
# model_g0_seed_101_other_101_10ep.pt
# model_g0_seed_201_other_201_10ep_lr5e-5.pt
# model_g0_seed_101_other_101_3ep.pt

# architecture
# model_g0_seed_201_other_201_5ep_small3.pt
# model_g0_seed_101_other_101_10ep_base.pt

# g2
# model_g2_seed_101_other_101_10ep.pt
# model_g2_seed_101_other_101_3ep.pt

init_checkpoint_path="${ch_dir}s1/model_g0_seed_201_other_201_10fps_10ep_base.pt"
prot_name='g0'
architecture='base'
other_id='10ep_30fps_base_adam'
seed=201
frame_rate=5

batch_size=64
num_workers=6

python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_toybox_embeddings.py -vid_root $vid_root -init_checkpoint_path $init_checkpoint_path -savedir $savedir --frame_rate $frame_rate --prot_name $prot_name --batch_size $batch_size --architecture $architecture --seed $seed --other_id $other_id --num_workers $num_workers 


#-------
init_checkpoint_path="${ch_dir}s1/model_g0_seed_201_other_201_15fps_10ep_base.pt"
prot_name='g2'
architecture='base'
seed=201
frame_rate=5
other_id='10ep_30fps_base_adam'

batch_size=64
num_workers=6

python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_toybox_embeddings.py -vid_root $vid_root -init_checkpoint_path $init_checkpoint_path -savedir $savedir --frame_rate $frame_rate --prot_name $prot_name --batch_size $batch_size --architecture $architecture --seed $seed --other_id $other_id --num_workers $num_workers 


# #-------
# init_checkpoint_path="${ch_dir}s1/model_g0_seed_101_other_101_10ep_base.pt"
# prot_name='g0'
# architecture='base'
# other_id='10ep_base'
# seed=101
# frame_rate=10

# batch_size=64
# num_workers=6

# python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_toybox_embeddings.py -vid_root $vid_root -init_checkpoint_path $init_checkpoint_path -savedir $savedir --frame_rate $frame_rate --prot_name $prot_name --batch_size $batch_size --architecture $architecture --seed $seed --other_id $other_id --num_workers $num_workers 
