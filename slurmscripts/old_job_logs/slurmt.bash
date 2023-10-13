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
#SBATCH --job-name=job_tgcu401
#SBATCH --output=job_tgcu401_Out
#SBATCH --error=job_tgcu401_Err

######  Module commands #####
module load deeplearning
ulimit -u 20000

seed=401

vid_root='/N/project/baby_vision_curriculum/benchmarks/toybox/vids/toybox/'
ch_dir='/N/project/baby_vision_curriculum/trained_models/generative/v2/may10/'
# model_g0_seed_1111_other_1111_mask50_small2_30ep.pt"
# model_g2_seed_1111_other_1111_mask50_small2_30ep.pt"
#g0g1g2: s3/model_g2_seed_2323_other_2233_mask50_small2_pre.g0g1.pt"
#g2g1g0: s3/model_g0_seed_2323_other_2233_mask50_small2_pre.g2g1.pt"
#g0g1: s2/model_g1_seed_1133_other_1345_mask50_small2_pre.g0.pt"
#g2g1: s2/model_g1_seed_1133_other_1345_mask50_small2_pre.g2.pt"
#g0g1
init_checkpoint_path="${ch_dir}s3/model_g2_seed_401_other_401_3ep_pre.g0g1.pt"
prot_name='g0g1g2'


savedir='/N/project/baby_vision_curriculum/trained_models/generative/v2/benchmarks/toybox/'
frame_rate=3
batch_size=128

other_id='10fps.3ep'

num_workers=6
architecture='small2'

python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_toybox_embeddings.py -vid_root $vid_root -init_checkpoint_path $init_checkpoint_path -savedir $savedir --frame_rate $frame_rate --prot_name $prot_name --batch_size $batch_size --architecture $architecture --seed $seed --other_id $other_id --num_workers $num_workers 


#-------
init_checkpoint_path="${ch_dir}s3/model_g0_seed_401_other_401_3ep_pre.g2g1.pt"
prot_name='g2g1g0'

python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_toybox_embeddings.py -vid_root $vid_root -init_checkpoint_path $init_checkpoint_path -savedir $savedir --frame_rate $frame_rate --prot_name $prot_name --batch_size $batch_size --architecture $architecture --seed $seed --other_id $other_id --num_workers $num_workers 

#-------
init_checkpoint_path="${ch_dir}s1/model_g0_seed_401_other_401_3ep.pt"
prot_name='g0'

python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_toybox_embeddings.py -vid_root $vid_root -init_checkpoint_path $init_checkpoint_path -savedir $savedir --frame_rate $frame_rate --prot_name $prot_name --batch_size $batch_size --architecture $architecture --seed $seed --other_id $other_id --num_workers $num_workers 

#-------
init_checkpoint_path="${ch_dir}s1/model_g2_seed_401_other_401_3ep.pt"
prot_name='g2'

python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_toybox_embeddings.py -vid_root $vid_root -init_checkpoint_path $init_checkpoint_path -savedir $savedir --frame_rate $frame_rate --prot_name $prot_name --batch_size $batch_size --architecture $architecture --seed $seed --other_id $other_id --num_workers $num_workers 

#-------
init_checkpoint_path="${ch_dir}s2/model_g1_seed_401_other_401_3ep_pre.g0.pt"
prot_name='g0g1'

python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_toybox_embeddings.py -vid_root $vid_root -init_checkpoint_path $init_checkpoint_path -savedir $savedir --frame_rate $frame_rate --prot_name $prot_name --batch_size $batch_size --architecture $architecture --seed $seed --other_id $other_id --num_workers $num_workers 


#-------
init_checkpoint_path="${ch_dir}s2/model_g1_seed_401_other_401_3ep_pre.g2.pt"
prot_name='g2g1'

python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/compute_toybox_embeddings.py -vid_root $vid_root -init_checkpoint_path $init_checkpoint_path -savedir $savedir --frame_rate $frame_rate --prot_name $prot_name --batch_size $batch_size --architecture $architecture --seed $seed --other_id $other_id --num_workers $num_workers 