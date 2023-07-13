import argparse
import datetime
import random
import time
from pathlib import Path
import itertools

import torch
import torchvision.transforms as standard_transforms
import numpy as np

from PIL import Image
import cv2
from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
import warnings
warnings.filterwarnings('ignore')

from util.loading_data import loading_data

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
    
    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    parser.add_argument('--data_root', default='./crowd_datasets/Unified-Crowd',
                        help='path where the dataset is')
    parser.add_argument('--output_dir', default='output',
                        help='path where to save')
    # parser.add_argument('--weight_path', default='weights/best_mae.pth',
    #                     help='path where the trained weights saved')
    parser.add_argument('--weight_path', default='weights/SHTechA.pth',
                        help='path where the trained weights saved')

    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')

    return parser

def main(args, debug=False):

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    print(args)
    device = torch.device('cuda')
    # get the P2PNet
    model = build_model(args)
    # move to GPU
    model.to(device)
    # load trained model
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    # convert to eval mode
    model.eval()

    _, val_set = loading_data(args.data_root)

    img, target = val_set.__getitem__(0)

    img = img.to(device)

    patch_size = 128
    stride = 128

    patches = img.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    h_patches = patches.shape[1]
    v_patches = patches.shape[2]
    patches = patches.permute(1, 2, 0, 3, 4).reshape(-1, 3, 128, 128)

    # for i, patch in enumerate(patches):
    #     patch = patch.detach().cpu().numpy()
    #     patch = np.transpose(patch, (1, 2, 0))
    #     # Denormalize
    #     mean = [0.485, 0.456, 0.406]
    #     std = [0.229, 0.224, 0.225]
    #     patch = patch * std + mean
    #     patch = (patch * 255).astype(np.uint8)
    #     img_to_draw = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite(os.path.join(args.output_dir, 'patch{}.jpg'.format(i)), img_to_draw)
    
    # run inference
    outputs = model(patches)
    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1].detach().cpu().numpy()

    print(outputs_scores.shape)

    outputs_points = outputs['pred_points'].detach().cpu().numpy()

    offsets = np.array(list(itertools.product(np.arange(h_patches) * stride, np.arange(v_patches) * stride)))
    print(offsets)
    offsets = np.flip(offsets, -1)
    print(offsets)
    offsets = np.repeat(offsets[:, np.newaxis, :], outputs_points.shape[1], axis=1)
    outputs_points = outputs_points + offsets

    threshold = 0.5
    # filter the predictions
    points = outputs_points[outputs_scores > threshold]
    predict_cnt = int((outputs_scores > threshold).sum())

    print(points)
    print(offsets[outputs_scores > threshold])
    print(np.transpose(np.nonzero(outputs_scores > threshold)))

    img = img.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    # Denormalize
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    img = img * std + mean

    img = (img * 255).astype(np.uint8)

    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

    outputs_points = outputs['pred_points'][0]
    # draw the predictions
    size = 2
    img_to_draw = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for p in points:
        img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
    # save the visualized image
    cv2.imwrite(os.path.join(args.output_dir, 'pred{}.jpg'.format(predict_cnt)), img_to_draw)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)