import sys, os

env_root = '/N/project/baby_vision_curriculum/pythonenvs/hfenv/lib/python3.10/site-packages/'
sys.path.insert(0, env_root)

# import numpy as np
import torch, torchvision
import cv2
from pathlib import Path
import warnings
import numpy as np
from torchvision import transforms as tr

def _get_transform(image_size):

    mean = [0.5, 0.5, 0.5]#np.mean(mean_all, axis=0) #mean_all[chosen_subj] 
    std = [0.25, 0.25, 0.25] #std_all[chosen_subj] 
    
#     [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
#     [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD

    augs = [tr.Resize(image_size), tr.CenterCrop(image_size), 
            tr.ConvertImageDtype(torch.float32), 
             tr.Normalize(mean,std)]
    return tr.Compose(augs)

def transform_vid(video):
    # Used with standard video datasets such as torchvision.UCF101
#     print(vid.shape)
    if video.shape[1]!=3: # Make it TCHW
        video = torch.permute(video, (0,3,1,2))
    image_size = 224
#     vid.p
    transform = _get_transform(image_size)
#     xtt = [transform(torch.from_numpy(frame)).unsqueeze(0) 
    xtt = [transform(frame).unsqueeze(0) 
       for frame in video]
    return torch.concat(xtt, axis=0)#.unsqueeze(0)

def transform_image(image, sample_len=16):
#     Used for standard single image datasets such as torchvision.CIFAR10, torchvision.ImageNet
#     if image.shape[0]!=3:
    image_size=224
    num_frames=sample_len
    transform = _get_transform(image_size)
    return transform(image).unsqueeze(0).repeat(num_frames,1,1,1)



class SSv2Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform, frame_rate=12, sample_len=16, train=True):
        if train:
            self.root_dir = os.path.join(root_dir, 'train/')
        else:
            self.root_dir = os.path.join(root_dir, 'val/')
        self.frame_rate = frame_rate
        self.sample_len = sample_len
        self.transform = transform
        self.samples = sorted(os.listdir(self.root_dir),
                             key=lambda x: int(x))
        self.fps = 12 #original frame rate
        self.ds_rate = round(self.fps/self.frame_rate)

    def __len__(self):
        return len(self.samples)
    
    def _read_frames(self, sample_dir, framefn_list):
        vid= [torchvision.io.read_image(
            str(Path(self.root_dir, sample_dir, fn))).unsqueeze(0)
                for fn in framefn_list]
        return torch.concat(vid, axis=0)
        
    def get_frames(self, sample_dir):
        framefn_list = sorted(os.listdir(self.root_dir+sample_dir),
                              key=lambda x: int(x.split('.')[0]))
        # try selecting with the suggested ds_rate, starting from the suggested point.
        # if not enough, try starting from the beginning
        # if still not enough, gradually reduce ds_rate.
        # if not enough frames for ds_rate=1, repeat the last frame.
        num_frames = len(framefn_list)
        loc_idx = num_frames//4
        slen = self.sample_len
        step = self.ds_rate
        if num_frames//step <slen:
            last_item = framefn_list[-1]
            while (len(framefn_list)//step)<slen:
                framefn_list.append(last_item)
            return self._read_frames(sample_dir, framefn_list[::step][:slen])
        elif (num_frames-loc_idx)//step <slen:
            return self._read_frames(sample_dir, framefn_list[::step][:slen])
        else:
            return self._read_frames(sample_dir, framefn_list[loc_idx:loc_idx+slen*step:step][:slen])


        
    def __getitem__(self, index):
#         print('---------------')
        vid_fname = self.samples[index]
        frames = self.get_frames(vid_fname)
#         if self.transform is not None:
#             frames = [self.transform(fr)
#                       for fr in frames]
        if self.transform is not None:
            frames = self.transform(frames)
        return frames, vid_fname
        
class ToyboxDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform, frame_rate=10, sample_len=16, train=True):
        self.root_dir = root_dir
        self.frame_rate = frame_rate
        self.sample_len = sample_len
        self.transform = transform
        self.samples = []
        for supercategory in os.listdir(self.root_dir):
            for obj in os.listdir(os.path.join(self.root_dir, supercategory)):
#                 for obj in os.listdir(os.path.join(self.root_dir, supercategory, category)):
                object_dir = os.path.join(self.root_dir, supercategory, obj)
                for view in os.listdir(object_dir):
                    view_path = os.path.join(object_dir, view)
                    self.samples.append(view_path)
#                         self.samples.append((view_path, supercategory, category, object))

    def __len__(self):
        return len(self.samples)

    def get_all_frames(self, cap):
        desired_frames = self.sample_len
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
#                 print('end of the video, i_frame, len fames', frame_count, len(frames))
                # End of video
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)                
            if len(frames) == desired_frames:
                break
        tmp_nframes = len(frames)
        if tmp_nframes < desired_frames:
            last_frame = frames[-1]
            for i in range(desired_frames - tmp_nframes):
                frames.append(last_frame)
        
        assert len(frames)==desired_frames
        return frames
    
    def wrap_frames(self, frames):
        frames = torch.as_tensor(np.asarray(frames))
        if len(frames.shape)!=4: #torch.Size([16, 12xx, 19xx, 3])
            return None
        return self.transform(frames)
            
    def __getitem__(self, index):
#         print('---------------')
        vid_path = self.samples[index]
        vid_fname = Path(vid_path).name
        frames = []
        cap = cv2.VideoCapture(vid_path)
        if cap is None or not cap.isOpened():
            warnings.warn('unable to open video source: '+vid_path)
            return None, None

        fps = cap.get(cv2.CAP_PROP_FPS)
        ds_rate = round(fps/self.frame_rate)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         print('num_frames:',num_frames)
#         print('ds_rate:',ds_rate)
#         print('num_frames:',num_frames)
        
        sample_scope = self.sample_len*ds_rate
        if num_frames<sample_scope:
#             print('Not enough frames in the video',vid_path)
            frames = self.get_all_frames(cap)
                        #apply transform
            frames_transformed = self.wrap_frames(frames)
            if frames_transformed is None:
                print(vid_path, 'gave None')
                return None, None
            return frames_transformed, vid_fname
            
        
        # duration = num_frames / fps
        start_frame = int(num_frames * 1 / 5)  # Starting frame at 2/3 of video duration
        if (num_frames-start_frame)<sample_scope:
            start_frame = num_frames-sample_scope
        
#         print('start_frame',start_frame)
#         end_frame = start_frame+sample_scope#int(start_frame + fps * 1.6)  # Ending frame after 1.6 seconds
        desired_frames = self.sample_len
        frame_count = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
#                 print('end of the video, i_frame, len fames', frame_count, len(frames))
                # End of video
                break
            
            if frame_count % ds_rate==0:
#                 if (frame_count > start_frame) & \
#                 (frame_count < end_frame):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)                
            if len(frames) == desired_frames:
                break
            frame_count += 1
            
        cap.release()
        frames_transformed = self.wrap_frames(frames)
        
        if frames_transformed is None:
            print(vid_path, 'gave None')
            return None, None
        else:
            return frames_transformed, vid_fname
        

def get_inp_label(task, batch):
    if task=='ucf101':
        inputs, _, labels = batch
        return inputs, labels
    else:
        raise NotImplementedError()

def ucf_collate(batch):
    filtered_batch = []
    for video, _, label in batch:
        filtered_batch.append((video, label))
    return torch.utils.data.dataloader.default_collate(filtered_batch)


class UCF101(torchvision.datasets.ucf101.UCF101):
    def __init__(self, root, annotation_path, frames_per_clip, step_between_clips=1, 
                 frame_rate= None, fold=1, train=True, transform=None, _precomputed_metadata=None, 
                 num_workers=1, _video_width=0, _video_height=0,
                 _video_min_dimension=0, _audio_samples=0, output_format='THWC'):
        super(UCF101, self).__init__(root, annotation_path, frames_per_clip, 
                                       step_between_clips=step_between_clips, 
                 frame_rate=frame_rate, fold=fold, train=train, transform=transform, 
                                       _precomputed_metadata=_precomputed_metadata, 
                 num_workers=num_workers, _video_width=_video_width, _video_height=_video_height,
                 _video_min_dimension=_video_min_dimension, _audio_samples=_audio_samples, 
                                       output_format=output_format)
        print('samples len:',len(self.samples))
        print('element type:',type(self.samples[0][0]))
        video_list = [x[0] for x in self.samples]
        
        self.indices = []
#         fold=1
        for train in [True, False]:
            fold=fold
#             for fold in range(1,4):
            self.indices+=self._select_fold(video_list, annotation_path, fold, train)
#         self.indices = self.indices[::]
        video_clips = self.full_video_clips
        self.video_clips = video_clips.subset(self.indices)
        self.transform = transform
        
def make_ucf101dataset(sample_len, frame_rate, num_workers=6, train=True):
    ucf_root='/N/project/baby_vision_curriculum/benchmarks/mainstream/ucf101/UCF-101'
    annotation_path = '/N/project/baby_vision_curriculum/benchmarks/mainstream/ucf101/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/'
    frames_per_clip = sample_len #16
    step_between_clips = 300#1
#     frame_rate=args.frame_rate#int(30/args.ds_rate)
    transform = transform_vid
    output_format= 'TCHW'
#     num_workers=args.num_workers-1 #40
    dataset = UCF101(ucf_root, 
                      annotation_path,
                      frames_per_clip,
                      step_between_clips=step_between_clips,
                      frame_rate=frame_rate,
                      fold=1,
                      train=train,
                      transform=transform,
                      output_format=output_format,
                      num_workers=num_workers)
    # Note: MyUCF101 collects both train and val split for the given fold
    # num_classes = 101
    return dataset#, num_classes
#     return {'train':train_dataset,
#            'val':val_dataset}, num_classes

class Cifar10Transform():
    def __init__(self, sample_len, image_size=224):
        self.sample_len = sample_len
        self.image_size= image_size
        
        mean = [0.5, 0.5, 0.5]
        std = [0.25, 0.25, 0.25]

        augs = [tr.ToTensor(),
                tr.Resize(image_size), tr.CenterCrop(image_size),
                tr.ConvertImageDtype(torch.float32), 
                tr.Normalize(mean, std)]
    
        self.transform_image = tr.Compose(augs)
        
    def __call__(self, image):
        return self.transform_image(image).unsqueeze(0).repeat(self.sample_len, 1, 1, 1)
        
        
# def transform_image_cifar10(image, sample_len):
#     # Your existing transformation logic here
#     image_size = 224
#     num_frames = sample_len
#     mean = [0.5, 0.5, 0.5]
#     std = [0.25, 0.25, 0.25]

#     augs = [tr.ToTensor(),
#             tr.Resize(image_size), tr.CenterCrop(image_size),
#             tr.ConvertImageDtype(torch.float32), 
#             tr.Normalize(mean, std)]
#     transform = tr.Compose(augs)

#     return transform(image).unsqueeze(0).repeat(num_frames, 1, 1, 1)

def make_cifar10dataset(sample_len, train=False):
    cifar10img_root = '/N/project/baby_vision_curriculum/benchmarks/mainstream/cifar10'
    dataset = torchvision.datasets.CIFAR10(root=cifar10img_root,
                               transform=Cifar10Transform(sample_len),
                               train=train, download=True)
    return dataset

# def make_cifar10dataset(sample_len):
#     cifar10img_root = '/N/project/baby_vision_curriculum/benchmarks/mainstream/cifar10'
    
#     def transform_image_cifar10(image):#, sample_len=sample_len):
#     #     Used for standard single image datasets such as torchvision.CIFAR10, torchvision.ImageNet
#     #     if image.shape[0]!=3:
#         image_size=224
#         num_frames=sample_len
#         mean = [0.5, 0.5, 0.5]#np.mean(mean_all, axis=0) #mean_all[chosen_subj] 
#         std = [0.25, 0.25, 0.25] #std_all[chosen_subj] 

#         augs = [tr.ToTensor(),
#                 tr.Resize(image_size), tr.CenterCrop(image_size),
#                 tr.ConvertImageDtype(torch.float32), 
#                  tr.Normalize(mean,std)]
#         transform = tr.Compose(augs)

#         return transform(image).unsqueeze(0).repeat(num_frames,1,1,1)
#     dataset = torchvision.datasets.CIFAR10(root=cifar10img_root,
#                                            transform=transform_image_cifar10, 
#                                            train=False, download=True)
#     return dataset
