import numpy as np
import cv2
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.patches as patches
import math
from utils import *
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter
import seaborn as sns
from mpl_toolkits.axes_grid1 import ImageGrid

def display_overlaid_image(image, joints_X, joints_Y, center):
    fig, ax = plt.subplots(1)
    fig.set_size_inches(18.5, 10.5)

    joints_X = np.array(joints_X[joints_X>0])
    joints_Y = np.array(joints_Y[joints_Y>0])

    plt.imshow(image)
    plt.scatter(center[0]*224, center[1]*224, marker='o', color='y', s=200) # plot center of the human
    plt.scatter(joints_X*224, joints_Y*224, color='r', s=100) # plot the joints
    plt.show()

def display_overlaid_image_dual(image, joints_X, joints_Y, joints_X_pred, joints_Y_pred, center, height, width, image_tag, visibility_array, heatmap_stack_real=None, heatmap_stack_pred=None):
    # image = np.array(image).astype(np.uint8)
    # image = (image * 0.5 + 0.5)
    # xs = joints_X * width
    # ys = joints_Y * height

    # fig, ax = plt.subplots(1)
    # fig.set_size_inches(10, 10)

    joints_X = np.array(np.where(visibility_array>0, joints_X, 0))
    joints_Y = np.array(np.where(visibility_array>0, joints_Y, 0))

    joints_X_pred = np.array(np.where(joints_X_pred > 0, joints_X_pred, 0))
    joints_X_pred = np.array(np.where(joints_Y_pred > 0, joints_X_pred, 0))


    joints_Y_pred = np.array(np.where(joints_X_pred > 0, joints_Y_pred, 0))
    joints_Y_pred = np.array(np.where(joints_Y_pred > 0, joints_Y_pred, 0))

    # joints_X_pred = np.array(joints_X_pred[joints_X_pred > 0])
    # joints_Y_pred = np.array(joints_Y_pred[joints_Y_pred > 0])

    save_path = get_save_path()
    save_filename = f'{image_tag}.png'
    save_path_full = os.path.join(save_path, save_filename)

    num_joints = len(joints_X)
    joints_X *= width
    joints_Y *= height

    joints_X_pred *= width
    joints_Y_pred *= height
    center[0] *= width
    center[1] *= height


    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1)

    for index in range(num_joints - 1):
        if index == 9 or index == 5:
            continue
        start_point = (joints_X[index], joints_X[index + 1])
        end_point = (joints_Y[index], joints_Y[index + 1])

        if start_point[0] > 0 and start_point[1] > 0 and end_point[0] > 0 and end_point[1] > 0:
            plt.plot(start_point, end_point, linewidth=5)
    ax1.imshow(image)
    ax1.scatter(center[0], center[1], marker='o', color='y', s=200)  # plot center of the human
    ax1.scatter(joints_X, joints_Y, color='b', marker='x', s=100)  # plot the joints (real)
    ax1.set_title('Ground Truth Joints')

    ax2 = fig.add_subplot(1, 2, 2)
    for index in range(num_joints - 1):
        if index == 9 or index == 5:
            continue
        start_point = (joints_X_pred[index], joints_X_pred[index + 1])
        end_point = (joints_Y_pred[index], joints_Y_pred[index + 1])

        if start_point[0] > 0 and start_point[1] > 0 and end_point[0] > 0 and end_point[1] > 0:
            plt.plot(start_point, end_point, linewidth=5)

    ax2.imshow(image)
    ax2.scatter(joints_X_pred, joints_Y_pred, color='r', s=100)  # plot the joints (pred)
    ax2.set_title('Predicted Joints')

    # ax3 = fig.add_subplot(2, 2, 3)
    # ax3.imshow(heatmap_real)
    # im = ax3.imshow(heatmap_real, cmap=plt.cm.RdBu, extent=(-3, 3, 3, -3), interpolation='bilinear')
    # plt.colorbar(im)
    # axc = sns.heatmap(heatmap_stack_real, vmin=0, vmax=1)

    # ax3.set_title('Ground Truth Heatmap')

    # ax4 = fig.add_subplot(2, 2, 4)
    # axd = sns.heatmap(heatmap_stack_pred, vmin=0, vmax=1)
    # ax4.set_title('Predicted Heatmap')

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    # ax3.set_xticks([])
    # ax3.set_yticks([])
    # ax4.set_xticks([])
    # ax4.set_yticks([])

    plt.savefig(save_path_full, bbox_inches='tight')
    plt.close('all')



def visualize_keypoint_heatmaps(heatmap_stack_real, heatmap_stack_coarse, heatmap_stack_fine, image, image_tag='demo'):
    save_path = get_save_path('heatmaps')
    save_filename = f'{image_tag}.png'
    save_path_full = os.path.join(save_path, save_filename)

    num_keypoints = heatmap_stack_fine.shape[0]

    rows = int(np.ceil(num_keypoints / 2))
    # plt.figure(figsize=(16, 2))
    # for index in range(num_keypoints):
    #     plt.subplot(rows, rows, index + 1)
    #     plt.axis('off')
    #     plt.imshow(image)
    #     plt.imshow(heatmap_stack[index], cmap='seismic', alpha=0.8)
    #     plt.title(index)
    # # plt.show()
    # plt.savefig(save_path_full)



    # fig = plt.figure(figsize=(10, 5))
    #
    # for index in range(num_keypoints):
    #     exec(f'ax = fig.add_subplot(16, 2, {index+1})')
    #
    #     ax.imshow(image)
    #     ax.imshow(heatmap_stack[index], cmap='seismic', alpha=0.8)
    #     ax.set_title(index)
    #     # exec(f'ax{index + 1} = fig.add_subplot(16, 2, {index + 1})')
    #     # exec(f'ax{index + 1}.imshow({image})')
    #     # exec(f'ax{index + 1}.imshow({heatmap_stack[index]}, cmap=\'seismic\', alpha=0.8)')
    #     # exec(f'ax{index + 1}.set_title({index})')
    # plt.savefig(save_path_full)
    heatmaps_combined = np.append(heatmap_stack_coarse, heatmap_stack_fine, axis=0)
    heatmaps_combined = np.append(heatmap_stack_real, heatmaps_combined, axis=0)
    fig = plt.figure(figsize=(24, 12))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(6, 8),  # creates 2x2 grid of axes
                     axes_pad=0.3,  # pad between axes in inch.
                     )
    heatmap_index = 0
    for index, (ax, heatmap) in enumerate(zip(grid, heatmaps_combined)):
        # Iterating over the grid returns the Axes.

        ax.imshow(image)
        ax.imshow(heatmap, cmap='seismic', alpha=0.7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(heatmap_index)
        heatmap_index += 1
        if heatmap_index == 16:
            heatmap_index = 0
    plt.savefig(save_path_full, bbox_inches='tight')
    plt.close('all')

def get_save_path(path='output'):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def get_np_image_from_tensor(tensor_image):
    np_image = tensor_image.permute(1, 2, 0).detach().cpu().numpy()
    return np_image

def get_np_from_tensor(tensor_array):
    np_array = tensor_array.detach().cpu().numpy()
    return np_array

def separate_X_Y(coordinates_combined):
    coordinates_combined = coordinates_combined.detach().cpu().numpy()
    joints_X = coordinates_combined[::2]
    joints_Y = coordinates_combined[1::2]

    return joints_X, joints_Y

class MPII_dataset(Dataset):
    
    def __init__(self, custom_transforms=None):
        dataset_path = os.path.join('data', 'mpii')
        annotations_path = os.path.join(dataset_path, 'annotations')
        self.images_path = os.path.join(dataset_path, 'images_cropped')
    
        self.custom_transforms = custom_transforms
        df_name = 'trainval_unique_cropped.csv'
        df_fullpath = os.path.join(annotations_path, df_name)
        self.df = pd.read_csv(df_fullpath)
        self.num_samples = len(self.df)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        current_sample = self.df.iloc[index] # row of dataframe
        filename = current_sample[0] # name of the image file

        full_path_image = os.path.join(self.images_path, filename) # full path to image file
        image = Image.open(full_path_image)

        # height, width, num_chanels = get_image_shape(image)
        height = self.df.iloc[index]['height']
        width = self.df.iloc[index]['width']

        center = np.array(current_sample[1:3]).astype(np.float32) # center of human
        scale = np.array(current_sample[3]).astype(np.float32) # scale of human

        joints_coordinates = np.array(current_sample[4:4+32]).astype(np.float32) # 16 joints (32 coordinates)
        visibility_array = np.array(current_sample[36:36+16]).astype(np.int64) # which joints are visible
        
        if self.custom_transforms is None:
            self.custom_transforms = transforms.Compose([
                transforms.ToTensor()
            ])
        
        image = self.custom_transforms(image)
        
        
        return image, torch.tensor(joints_coordinates), torch.tensor(visibility_array), torch.tensor(height), torch.tensor(width), torch.tensor(center)


def apply_sigmoid(image):
    return 1 / (1 + np.exp(-image))


def normalize(image, min_value=0, max_value=1):
    return (image - min_value) / (max_value - min_value)


def generate_keypoint_heatmap(image, height_batch, width_batch, visibility_array_batch, coordinates_real, sigma=5):
    height = int(height_batch[0])
    width = int(width_batch[0])
    batch_size = image.shape[0]
    num_joints = visibility_array_batch.shape[1]
    heatmap_stack = np.zeros((batch_size, num_joints, height, width)).astype(np.float32)

    for image_index in range(batch_size):
        visibility_array = visibility_array_batch[image_index]
        current_coordinates = coordinates_real[image_index]

        joints_X_real, joints_Y_real = separate_X_Y(current_coordinates * height)
        joints_X_real = np.array(np.where(visibility_array > 0, joints_X_real, -1))
        joints_Y_real = np.array(np.where(visibility_array > 0, joints_Y_real, -1))

        for joint_index in range(num_joints):
            current_Y = int(joints_Y_real[joint_index])
            current_X = int(joints_X_real[joint_index])

            if not (current_X >= height or current_X < 0 or current_Y >= width or current_Y < 0):
                heatmap_stack[image_index, joint_index, current_Y, current_X] = 1
                heatmap_stack[image_index, joint_index] = gaussian_filter(heatmap_stack[image_index, joint_index], sigma=sigma)
                heatmap_stack[image_index, joint_index] *= (1 / heatmap_stack[image_index, joint_index].max())

    heatmap_stack = torch.tensor(heatmap_stack)
    return heatmap_stack


def get_max_value_heatmap(heatmap_stack):
    # heatmap_stack.shape = batch_size x num_keypoints x height x width
    batch_size = heatmap_stack.shape[0]
    num_keypoints = heatmap_stack.shape[1]

    keypoints_batch = torch.zeros((batch_size, num_keypoints * 2), dtype=torch.float32)

    for batch_index in range(batch_size):
        filter_index = 0
        for keypoint_index in range(0, num_keypoints*2, 2):
            row, col = torch.where(heatmap_stack[batch_index, filter_index] == torch.max(heatmap_stack[batch_index, filter_index]))
            if col.size()[0] > 0 or row.size()[0] > 0:
                row = row[-1]
                col = col[-1]
            keypoints_batch[batch_index, keypoint_index] = col
            keypoints_batch[batch_index, keypoint_index + 1] = row
            filter_index += 1

    return keypoints_batch

def load_pretrained_model(load_checkpoint, checkpoint_name, checkpoint_path, model, optimizer, epoch_start, loss_train_best):
    if load_checkpoint:
        checkpoint_filename = f'{checkpoint_name}.pth'
        checkpoint_fullpath = os.path.join(checkpoint_path, checkpoint_filename)
        assert os.path.exists(checkpoint_fullpath), ('checkpoint do not exits for %s' % checkpoint_path)

        checkpoint_saved = torch.load(checkpoint_fullpath, map_location='cpu')

        model.load_state_dict(checkpoint_saved['model_state_dict'])
        optimizer.load_state_dict(checkpoint_saved['optimizer_state_dict'])
        epoch_start = checkpoint_saved['epoch_index'] + 1
        loss_train_best = checkpoint_saved['best_loss']
        print(f'Checkpoint loaded: {checkpoint_filename}')

    return model, optimizer, epoch_start, loss_train_best