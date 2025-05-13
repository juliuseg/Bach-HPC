import Skeleton_model.No_Warn
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import copy
from glob import glob
import gc
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from timeit import default_timer as timer
import random

from MiccaiModel.utils import get_config_from_json
from MiccaiModel.backbone.unet3d import UNet_ODT


import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
import pandas as pd
import napari
from scipy.spatial import distance_matrix
from skimage.draw import line_nd
import skimage.io as skio
import nibabel as nib
import numpy as np


# # Load AmiraGlow colormap
# from scipy import io
# cm_mat = io.loadmat('./weights/AmiraHotmap.mat')
# from matplotlib.colors import ListedColormap
# amiraglow = ListedColormap(cm_mat['AmiraHot'])

# colormap_array = np.concatenate((cm_mat['AmiraHot'], np.ones((256, 1))), axis=1)
# cm_napari_amiraglow = napari.utils.Colormap(
#         colors=colormap_array, display_name='AmiraGlow')

import torch

print("Loading model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Execution on device: {device}")

model = UNet_ODT()
model_path = './MiccaiModel/weights/model_weights_best.pth'

# Map the model to the correct device
model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
model = model.to(device)
model.eval()

config = get_config_from_json('MiccaiModel/config_unet.json')


print("Model loaded, now laoading data")

# Load dataset

# Step 1: Load the NIfTI file
nii_file = "/work3/s204427/NarwhalData/broken_segmentation.nii"
data = nib.load(nii_file)

size = 256
start_pos = 500
data_array = np.array(data.dataobj[start_pos:start_pos+size, 
                                   start_pos:start_pos+size, 
                                   start_pos:start_pos+size])

# binarize the data
data_array[data_array > 0.5] = 1
data_array[data_array <= 0.5] = 0

#print data array shape
print ("data array shape: ", data_array.shape)

# Create skeleton
data_array_skeleton = skeletonize(data_array)

# Create data for model
# make array of size 1,3,128,128,128
array = np.zeros((1,3,256,256,256))
array[0,1] = data_array
array[0,2] = data_array_skeleton
image_fused_patch = array

print ("data array shape: ", data_array.shape)
print ("image fused patch shape: ", image_fused_patch.shape)

# Normalize the data
for i in range(image_fused_patch.shape[1]):
    image_fused_patch[0, i] = (image_fused_patch[0, i] - np.min(image_fused_patch[0, i])) / (np.max(image_fused_patch[0, i]) - np.min(image_fused_patch[0, i]))

# rotate data
image_fused_patch[0,0] = np.rot90(image_fused_patch[0,0], axes=(1,2), k=1)
image_fused_patch[0,1] = np.rot90(image_fused_patch[0,1], axes=(1,2), k=1)
image_fused_patch[0,2] = np.rot90(image_fused_patch[0,2], axes=(1,2), k=1)

# Running the model
print ("Running the model")
torch.manual_seed(20221027)

# cuda exicution

with torch.no_grad():
    X_input = torch.from_numpy(image_fused_patch[:, 2:3]).to(device).float()  # Ensure float32 type

    output_skel = model(X_input)
    output_skel = output_skel.detach().cpu().numpy()

print ("Post processing the output")
# Post-process the output
idx_patch = 0
pred_mk3_mask = output_skel[idx_patch, 0].astype(np.float32)>0.5
pred_mk3_maskonly = np.logical_and(
    pred_mk3_mask, np.logical_not(image_fused_patch[idx_patch, 2].astype(np.bool_)))
pred_mk3 = output_skel[idx_patch, 0] * pred_mk3_maskonly

# binarize the pred_mk3
pred_mk3[pred_mk3 > 0.5] = 1
pred_mk3[pred_mk3 <= 0.5] = 0

# Save the output as non skeletonized, skeletonized, pred_mk3
# Construct first the np array
output_array = np.zeros((4, size,size,size))
output_array[0] = image_fused_patch[idx_patch, 1]
output_array[1] = image_fused_patch[idx_patch, 2]
output_array[2] = pred_mk3.astype(np.float32)
output_array[3] = skeletonize(pred_mk3.astype(np.float32))

# Save the output as .npy
random_id = random.randint(100000, 1000000)
save_dir = "MiccaiModel/plots"
os.makedirs(save_dir, exist_ok=True)  # Create directory
np.save(os.path.join(save_dir, f"output_{random_id}.npy"), output_array)
print (f"Output saved as {os.path.join(save_dir, f'output_{random_id}.npy')}")