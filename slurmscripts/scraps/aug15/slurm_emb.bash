#!/bin/bash
#####  Constructed by HPC everywhere #####
#SBATCH --mail-user=sheybani@iu.edu
#SBATCH -A r00117
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=0-02:00:00
#SBATCH --partition=gpu
#SBATCH --gpus 4
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --job-name=job_e2
#SBATCH --output=job_e2_Out
#SBATCH --error=job_e2_Err

######  Module commands #####
module load python/gpu
ulimit -u 20000

folder_id='aug11'
seed=153

# chpt_folder='aug11'
checkpoint_dir="/N/project/baby_vision_curriculum/trained_models/generative/v3/${folder_id}/"

ssv2_root='/N/project/baby_vision_curriculum/benchmarks/ssv2/easy_frames/'
ucf_root='/notUsed/'

ucsavedir="/N/project/baby_vision_curriculum/trained_models/generative/v3/${folder_id}/benchmarks/ucf101/"
sssavedir="/N/project/baby_vision_curriculum/trained_models/generative/v3/${folder_id}/benchmarks/ssv2/"


ucframe_rate=10
ssframe_rate=6
ucbatch_size=64
ssbatch_size=64
architecture='base'
num_workers=6
tubelet_size=2
num_frames=$((8*tubelet_size))



dataset_split='train'

init_checkpoint_path='notUsed'
run_id='notUsed'

# ds_task='ssv2'
# python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_videomae.py -ds_task $ds_task -vid_root $ssv2_root -init_checkpoint_path $init_checkpoint_path -savedir $sssavedir --frame_rate $ssframe_rate --run_id $run_id --batch_size $ssbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size --dataset_split $dataset_split --checkpoint_dir $checkpoint_dir

# ds_task='ucf101'
# python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_videomae.py -ds_task $ds_task -vid_root $ucf_root -init_checkpoint_path $init_checkpoint_path -savedir $ucsavedir --frame_rate $ucframe_rate --run_id $run_id --batch_size $ucbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size --dataset_split $dataset_split --checkpoint_dir $checkpoint_dir



dataset_split='test'

ds_task='ucf101'
python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_embeddings_videomae.py -ds_task $ds_task -vid_root $ucf_root -init_checkpoint_path $init_checkpoint_path -savedir $ucsavedir --frame_rate $ucframe_rate --run_id $run_id --batch_size $ucbatch_size --architecture $architecture --seed $seed --num_workers $num_workers --num_frames $num_frames --tubelet_size $tubelet_size --dataset_split $dataset_split --checkpoint_dir $checkpoint_dir