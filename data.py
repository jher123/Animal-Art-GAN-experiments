import torchvision.datasets as dset # for ImageFolder
import torch.utils.data
from torchvision import transforms, utils

from fastai.dataset import *

# This is just a template script summarising image data preparation for training

im_size = 64
root_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'data'
)
batch_size = 128
num_workers = 4 # number of data loading workers

# 1. Define transforms
# torchvision implements a lot of useful transforms
# alternatively we can implement custom transform class. They need to have __call__
# method so that they can be callable classes and the paramaters of the transform
# need not be passed every time it's called.
transforms = transforms.Compose(
    [transforms.Resize(im_size),
    transforms.CenterCrop(im_size),
    transforms.ToTensor(), # this swaps colour axis as numpy image is H x W x C and torch image is C x H x W, and does torch.from_numpy(image)
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] # mean and std of the images of each color channel
)

# fast.ai lib has a get_transforms function which returns a set of transforms that work generally well in a wide range of tasks
# It returns a tuple, one set for training set and one for validation
tfms =  get_transforms(
        do_flip=True, # if True random flip is applied with prob 0.5
        flip_vert=True, # if Trye the image can be flipped vertically or rotated by 90 degrees
        max_rotate=10, # maximum rotation, if None, random
        max_zoom=1.1,
        max_lighting=0.2,
        max_warp=0.2,
        p_affine=0.75, # the prob that each affine transform and symmetric warp is applied
        p_lighting=0.75, # as above for lighting transforms
        xtra_tfms=None
        )
)

# Alternatively, can use this method: Random center crop, pad_mode=cv2.BORDER_REFLECT
tfms = tfms_from_stats(inception_stats, IM_SIZE) # inception stats= ([[0.5, 0.5, 0.5 ], [0.5, 0.5, 0.5 ]])


# 2. Create a dataset object.
# We can use torchvision.datasets.ImageFolder which requires files to be organised in the format root/class_name/...jpg
# Or we can implement a custom dataset class inheriting from torch.utils.data.Dataset.
# It needs to have __len__ method returning the length of the dataset and __get_item__ returning
# one sample.
dataset = dset.ImageFolder(
            root=root_path,
            transforms=transforms
)

# 3. Create the dataloader: we need this to do batching, mutlithreading and shuffling
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# fastai version
# Create a model data object. The following example is from_csv which means we provide a csv with image label associations
md = ImageClassifierData.from_csv(
    PATH, img_folder_name, CSV_PATH, tfms=tfms, bs=BATCH_SIZE, skip_header=False, continuous=True
)

real_batch = next(iter(dataloader))
real_batch_fai = next(iter(md.trn_dl)) # training dataloader md.val_dl is validation dataloader

print(real_batch[0].size) # images
print(real_batch[1].size) # labels
