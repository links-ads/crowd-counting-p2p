import os
import glob
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2

class UnifiedDataset(Dataset):
    def __init__(self, data_root, transform=None, dataset="", train=False):
        if train:
            self.root_path = os.path.join(data_root, 'train')
        else:
            self.root_path = os.path.join(data_root, 'val')

        self.samples_list = sorted(glob.glob(os.path.join(self.root_path, f'{dataset}*')))
        
        self.nSamples = len(self.samples_list)
        
        self.transform = transform
        self.train = train
    
    @staticmethod
    def __load_data__(img_path, gt_path):
        # load the images
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # load ground truth points
        points = np.load(gt_path)

        return img, points

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        sample_path = self.samples_list[index]
        img_path = os.path.join(sample_path, 'img.jpg')
        gt_path = os.path.join(sample_path, 'points.npy')
        # load image and ground truth
        img, points = self.__load_data__(img_path, gt_path)
        # apply augumentation
        if self.transform is not None:
            transformed = self.transform(image=img, keypoints=points)
            img = transformed['image']
            points = transformed['keypoints']

        img = torch.Tensor(img)
        if len(points) > 0:
            points = torch.Tensor(points).float()
        else:
            points = torch.empty((0, 2))

        image_id = int(Path(sample_path).stem.split('_')[-1])
        if Path(sample_path).stem.split('_')[0] == 'NWPU':
            image_id += 5000
        image_id = torch.Tensor([image_id]).long()

        # SHHA Dataloader Adaptation
        target = [{'point': points, 'image_id': image_id, 'labels': torch.ones(points.shape[0]).long()}]


        return img, target