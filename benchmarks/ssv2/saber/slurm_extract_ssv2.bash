#!/bin/bash
#####  Constructed by HPC everywhere #####
#SBATCH --mail-user=sheybani@iu.edu
#SBATCH -A r00117
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:30:00
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --job-name=job_ss
#SBATCH --output=job_ss_Out
#SBATCH --error=job_ss_Err
module load python/gpu

vid_root='/N/project/baby_vision_curriculum/benchmarks/ssv2/20bn-something-something-v2/'
save_root='/N/project/baby_vision_curriculum/benchmarks/ssv2/easy_frames/val/'
easy_annot_path='/N/project/baby_vision_curriculum/benchmarks/ssv2/easy_labels/val_easy10.csv'

python /N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/ssv2/saber/extract_ssv2easy.py -vid_root $vid_root -save_root $save_root --easy_annot_path $easy_annot_path