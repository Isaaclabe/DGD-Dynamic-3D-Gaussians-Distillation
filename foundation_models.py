import os
import sys
import requests
import numpy as np
import uuid
from random import randint
from PIL import Image
import cv2
import clip

import torch
from torchvision import transforms
import torch.nn.functional as F

from lseg_minimal.lseg import LSegNet



class DINOv2_feature_extractor():
    def __init__(self, image_name, model_dinov2_net, image = None):
        self.image_name = image_name
        if image is not None:
          self.image = image
        else:
          self.image = Image.open(str(self.image_name)).convert('RGB')
        (self.H, self.W, self.d) = np.asarray(self.image).shape
        self.feature_dim_dinov2 = 384
        self.model_dinov2_net = model_dinov2_net
        self.patch_size = model_dinov2_net.patch_size
        self.image_tensor = self.process_image()
        self.patch_h  = self.H // self.patch_size
        self.patch_w  = self.W // self.patch_size

    def closest_mult_shape(self, n_H, n_W):
        closest_multiple_H = n_H * round(self.H // n_H)
        closest_multiple_W = n_W * round(self.W // n_W)
        return (closest_multiple_H, closest_multiple_W)

    def process_image(self):
        transform1 = transforms.Compose([
            transforms.Resize((self.H, self.W)),
            transforms.CenterCrop(self.closest_mult_shape(self.patch_size, self.patch_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.2)
        ])
        image_tensor = transform1(self.image).to('cuda').unsqueeze(0)
        return image_tensor

    def extract_feature(self):
        with torch.no_grad():
            # Extract per-pixel DINO features (384, H // patch_size, W // patch_size)
            image_feature_dino = self.model_dinov2_net.forward_features(self.image_tensor)['x_norm_patchtokens']
            image_feature_dino = image_feature_dino[0].reshape(self.patch_h, self.patch_w, self.feature_dim_dinov2)
        return image_feature_dino


class Lseg_feature_extractor():
    def __init__(self, image_name, model_lseg_net, image = None):
        self.image_name = image_name
        if image is not None:
          self.image = image
        else:
          self.image = cv2.cvtColor(cv2.imread(str(self.image_name)), cv2.COLOR_BGR2RGB)
        (self.H, self.W, self.d) = np.shape(self.image)
        self.image_tensor = self.process_image()
        self.feature_dim_lseg = 512
        self.model_lseg_net = model_lseg_net

    def process_image(self):
        image_tensor = cv2.resize(self.image, (int(2.23125*480), self.H))
        image_tensor = torch.from_numpy(image_tensor).float() / 255.0
        image_tensor = image_tensor[..., :3]  # drop alpha channel, if present
        image_tensor = image_tensor.cuda().permute(2, 0, 1).unsqueeze(0)  # 1, C, H, W
        return image_tensor

    def extract_feature(self):
        with torch.no_grad():
            # Extract per-pixel CLIP features (512, H // 2, W // 2)
            image_feature_clip = self.model_lseg_net.forward(self.image_tensor)[0].permute(1,2,0)

        return image_feature_clip
