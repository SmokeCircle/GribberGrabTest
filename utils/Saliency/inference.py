from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import argparse
import cv2
import numpy as np
import yaml
import os
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import SODModel
from .dataloader import InfDataloader, SODLoader
from .mask_yaml import MaskDetector

def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameters to train your model.')
    parser.add_argument('--imgs_folder', default='./data/DUTS/DUTS-TE/DUTS-TE-Image', help='Path to folder containing images', type=str)
    parser.add_argument('--model_path', default='/home/tarasha/Projects/sairajk/saliency/SOD_2/models/0.7_wbce_w0-1_w1-1.12/best_epoch-138_acc-0.9107_loss-0.1300.pt', help='Path to model', type=str)
    parser.add_argument('--use_gpu', default=True, help='Whether to use GPU or not', type=bool)
    parser.add_argument('--img_size', default=256, help='Image size to be used', type=int)
    parser.add_argument('--bs', default=24, help='Batch Size for testing', type=int)
    parser.add_argument('--use_SOD', action='store_true', default=False, help='whether to generate mask yaml file')

    return parser.parse_args()


def run_inference(args):
    # Determine device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(device='cuda')
    else:
        device = torch.device(device='cpu')

    # Load model
    model = SODModel()
    chkpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(chkpt['model'])
    model.to(device)
    model.eval()

    inf_data = InfDataloader(img_folder=args.imgs_folder, target_size=args.img_size)
    # Since the images would be displayed to the user, the batch_size is set to 1
    # Code at later point is also written assuming batch_size = 1, so do not change
    inf_dataloader = DataLoader(inf_data, batch_size=1, shuffle=True, num_workers=2)


    assert(os.path.isfile(args.imgs_folder))
    with torch.no_grad():
        for batch_idx, (img_np, img_tor, rotate) in enumerate(inf_dataloader, start=1):
            img_tor = img_tor.to(device)
            pred_masks, _ = model(img_tor)
            pred_masks_raw = np.squeeze(pred_masks.cpu().numpy(), axis=(0, 1))
            pred_masks_round = np.squeeze(pred_masks.round().cpu().numpy(), axis=(0,1))

            if rotate:
                pred_masks_raw = cv2.rotate(pred_masks_raw, cv2.ROTATE_90_COUNTERCLOCKWISE)
                pred_masks_round = cv2.rotate(pred_masks_round, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # remove little connections between areas
            kernel = np.ones((3, 3), np.uint8)
            pred_masks_round = cv2.erode(pred_masks_round, kernel, iterations=1)

            m = MaskDetector(pred_masks_round.astype(np.uint8), args.img_size, args.imgs_folder)
            m.write_yaml("./data/yaml/mask.yaml")
            m.write_img("./data/yaml/mask.png")

            if args.debug:
                print('Image :', batch_idx)
                img_np = np.squeeze(img_np.numpy(), axis=0)
                img_np = img_np.astype(np.uint8)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                if rotate:
                    img_np = cv2.rotate(img_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
                #cv2.imshow('Input Image', img_np)
                #cv2.imshow('Generated Saliency Mask', pred_masks_raw)
                #cv2.imshow('Rounded-off Saliency Mask', pred_masks_round)
                #key = cv2.waitKey(0)
                #if key == ord('q'):
                #    break


if __name__ == '__main__':
    rt_args = parse_arguments()
    run_inference(rt_args)
