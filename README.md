# baby-vision-curriculum


Self-supervised Pretraining on Infant Egocentric Videos

Experiments: 
- The effect of age-related curriculum
- The effect of spatial and temporal simplicity
- The effect of learning algorithm


Dependencies:
pytorch, torchvision
huggingface transformers (for VideoMAE)
tqdm

VideoMAE models need to be pretrained on multiple GPUs as they take up substantial GPU memory.
We use PyTorch DistributedDataParallel for that.
