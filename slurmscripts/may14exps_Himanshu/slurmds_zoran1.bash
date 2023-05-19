#!/bin/bash
#####  Constructed by HPC everywhere #####
#SBATCH --mail-user=hhansar@iu.edu
#SBATCH -A r00117
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=50
#SBATCH --time=0-20:00:00
#SBATCH --partition=gpu
#SBATCH --gpus 4
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --job-name=job_grrr_212
#SBATCH --output=job_grrr_212_Out
#SBATCH --error=job_grrr_212_Err

######  Module commands #####
module load deeplearning
ulimit -u 20000


task='imagenet'
ch_dir='/N/project/baby_vision_curriculum/trained_models/generative/v2/may14/'
# model_g0_seed_212_other_212_sgd_pre.g2g1.pt
# model_gr_seed_212_other_212_sgd_pre.grgr.pt
# model_g2_seed_212_other_212_sgd_pre.g3g1.pt
# model_g2_seed_212_other_212_sgd_pre.g0g1.pt
seed=212
prot_name='grgrgr'
init_checkpoint_path="${ch_dir}s3/model_gr_seed_212_other_212_sgd_pre.grgr.pt"



savedir='/N/project/baby_vision_curriculum/trained_models/generative/v2/benchmarks/imagenet/may14/'
finetune='n'
batch_size=64

other_id='sgd'

n_epoch=30
save_model='y'

frame_rate=30
num_workers=6
architecture='base'
lr=1e-3

python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/downstream_videomae_imagenet.py -task $task -init_checkpoint_path $init_checkpoint_path -savedir $savedir --save_model $save_model --frame_rate $frame_rate --prot_name $prot_name --lr $lr --batch_size $batch_size --architecture $architecture --n_epoch $n_epoch --seed $seed --other_id $other_id --num_workers $num_workers --finetune $finetune
