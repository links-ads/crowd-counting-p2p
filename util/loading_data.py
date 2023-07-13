import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.transforms import Normalize
from albumentations import HorizontalFlip, VerticalFlip
from albumentations.augmentations.crops.transforms import RandomCrop, CenterCrop
from albumentations.augmentations.geometric.resize import RandomScale, SmallestMaxSize
from util.augs import ConditionalSmallestMaxSize, ConditionalLongestMaxSize, ResizeMultiple
from util.unified_dataset import UnifiedDataset

def loading_data(data_root):
    # the pre-proccssing transform
    train_transform = A.Compose([
        RandomScale(scale_limit=0.3, always_apply=True),
        ConditionalSmallestMaxSize(max_size=128, always_apply=True),
        RandomCrop(height=128, width=128, always_apply=True),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
        ToTensorV2()
        ], 
        keypoint_params=A.KeypointParams(format='xy'))
    
    val_transform = A.Compose([
        ConditionalLongestMaxSize(max_size=1920, always_apply=True),
        ResizeMultiple(height=128, width=128, always_apply=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
        ToTensorV2()
        ], 
        keypoint_params=A.KeypointParams(format='xy'))
    # create the training dataset
    train_set = UnifiedDataset(data_root, train=True, transform=train_transform)
    # create the validation dataset
    val_set = UnifiedDataset(data_root, train=False, transform=val_transform)

    return train_set, val_set
