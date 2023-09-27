in_train_root = '/N/project/baby_vision_curriculum/benchmarks/mainstream/imagenet/ILSVRC/Data/CLS-LOC/train/'
# dset = torchvision.datasets.ImageNet(in_root, split='val')

# Adapted from https://www.kaggle.com/code/tusonggao/create-imagenet-train-subset-100k

def get_all_files(directory):
    filenames = []
    class_name = directory.split('/')[-1]
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            filenames.append(filename)
            part_name = filename.split('_')[0]
            assert part_name == class_name
    return filenames

train_path = in_train_root
# new_train_path = '/kaggle/working/imagenet_subtrain/'

filenames = []

dirs = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
print('len(dirs): ', len(dirs))

number_per_class = 100

for i, directory in enumerate(tqdm(dirs)):
    directory_path = train_path + directory
    filenames = get_all_files(directory_path)
    filenames = random.sample(filenames, number_per_class)
    assert len(filenames)==number_per_class
    
    # write the filenames and targets into one column.
    # make sure the target is the right imagenet index (how to go from synsets to indices).