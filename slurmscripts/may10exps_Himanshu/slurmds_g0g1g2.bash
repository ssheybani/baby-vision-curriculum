#!/bin/bash
#####  Constructed by HPC everywhere #####
#SBATCH --mail-user=hhansar@iu.edu
#SBATCH -A r00117
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=50
#SBATCH --time=0-15:00:00
#SBATCH --partition=gpu
#SBATCH --gpus 4
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --job-name=job_g012401
#SBATCH --output=job_g012401_Out
#SBATCH --error=job_g012401_Err

######  Module commands #####
module load deeplearning
ulimit -u 20000

seed=401

task='cifar10'
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


savedir='/N/project/baby_vision_curriculum/trained_models/generative/v2/benchmarks/cifar10/'
finetune='n'
batch_size=64

other_id='10fps.3ep'

n_epoch=5
save_model='n'

frame_rate=10
num_workers=6
architecture='small2'
lr=1e-3

python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/downstream_videomae_Himanshu.py -task $task -init_checkpoint_path $init_checkpoint_path -savedir $savedir --save_model $save_model --frame_rate $frame_rate --prot_name $prot_name --lr $lr --batch_size $batch_size --architecture $architecture --n_epoch $n_epoch --seed $seed --other_id $other_id --num_workers $num_workers --finetune $finetune
