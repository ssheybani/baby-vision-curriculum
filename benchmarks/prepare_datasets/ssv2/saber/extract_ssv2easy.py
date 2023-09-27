"""
Report downstream performance scores for a pretrained model.
"""
import sys, os

env_root = '/N/project/baby_vision_curriculum/pythonenvs/hfenv/lib/python3.10/site-packages/'
sys.path.insert(0, env_root)

# import numpy as np
from tqdm import tqdm
from pathlib import Path
# import math
import argparse
import pandas as pd
# import warnings
import cv2

def extract_frames(fp, save_dir):
    cap = cv2.VideoCapture(fp)
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        save_path = os.path.join(save_dir,
                         str(i)+'.jpg')
#         print(save_path)
        cv2.imwrite(save_path, frame)
        i+=1
    cap.release()
    return

# savedir = '/N/project/baby_vision_curriculum/benchmarks/ssv2/easy_frames/'
# 

    
#____________________________________________
if __name__ == '__main__':
    
    #---------- Parse the arguments
    parser = argparse.ArgumentParser(description='Evaluate downstream performance for a pretrained model.')

    # Add the arguments

    parser.add_argument('-vid_root',
                           type=str,
                           help='absolute path to the ssv2 webm videos root')
    
    parser.add_argument('-save_root',
                           type=str,
                           help='')
    
    parser.add_argument('--easy_annot_path',
                           type=str,
                           default='/N/project/baby_vision_curriculum/benchmarks/ssv2/easy_labels/train_easy10.csv',
                           help='')
    
    args = parser.parse_args()
    
    
    df = pd.read_csv(args.easy_annot_path)
    
    for fn in tqdm(df['fname']):
        fp = args.vid_root+fn
        save_dir = args.save_root+fn.split('.')[0]
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        extract_frames(fp, save_dir)
        