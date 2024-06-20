import os
import sys
import requests
import numpy as np
import uuid
from tqdm import tqdm
from random import randint
from PIL import Image
import cv2
import clip
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from utils.loss_utils import l1_loss, ssim, kl_divergence
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, get_linear_noise_func
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams


def plot_and_print_color(gaussians, image_color, gt_image, iteration):
    image_np = image_color.detach().cpu().permute(1, 2, 0).numpy()
    image_gt = gt_image.detach().cpu().permute(1, 2, 0).numpy()
    
    Number_points = np.shape(gaussians.get_xyz)[0]
    current_allocated = torch.cuda.memory_allocated()
    
    print("#------------#")
    print("Iteration:", iteration)
    print(f"Current GPU memory allocated: {current_allocated/(1024**3):.2f} GB")
    print("Number of points:", Number_points)
    print("#------------#")

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].imshow(image_np)
    axs[0].set_title("Original Image")
    axs[1].imshow(image_gt)
    axs[1].set_title("Image gt")
#    plt.savefig("/sci/labs/sagieb/isaaclabe/4D-gaussian-splatting-sementic-New-CLIP/Image_save_7/Image_" + str(iteration) + ".png")
    plt.show()


def plot_and_print_feature(gaussians, image_feature_clip, image_color, gt_image, iteration):
    pca = PCA(n_components = 3)
    PCA_array = image_feature_clip.permute(1, 2, 0).detach().cpu().numpy().reshape(image_feature_clip.shape[1]*image_feature_clip.shape[2], image_feature_clip.shape[0])
    pca.fit(PCA_array)
    pca_features = pca.transform(PCA_array)
    for i in range(3):
        pca_features[:, i] = (pca_features[:, i] - pca_features[:, i].min()) / (pca_features[:, i].max() - pca_features[:, i].min())
    image_np = image_color.detach().cpu().permute(1, 2, 0).numpy()
    image_gt = gt_image.detach().cpu().permute(1, 2, 0).numpy()
    
    Number_points = np.shape(gaussians.get_xyz)[0]
    print(Number_points)
    current_allocated = torch.cuda.memory_allocated()
    
    print("#---------#")
    print("Iteration:", iteration)
    print(f"Current GPU memory allocated: {current_allocated / (1024 ** 3):.2f} GB")
    print("Number of points:",Number_points)
    print("#---------#")

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image_np)
    axs[0].set_title("Original Image")
    axs[1].imshow(image_gt)
    axs[1].set_title("Image gt")
    axs[2].imshow(pca_features.reshape(image_feature_clip.shape[1], image_feature_clip.shape[2], 3))
    axs[2].set_title("PCA Features")
    plt.subplots_adjust(wspace=0.3)
#    plt.savefig("/sci/labs/sagieb/isaaclabe/4D-gaussian-splatting-sementic-New-CLIP/Image_save_7/Image_" + str(iteration) + ".png")
    plt.show()


def freeze_grad(gaussians, name, state = True):
    for param_group in gaussians.optimizer.param_groups:
        if param_group['name'] == name:
            for param in param_group['params']:
                param.requires_grad = state
