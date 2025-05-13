import numpy as np
from Skeleton_model.st3d import structure_tensor, eig_special
from scipy.ndimage import label, binary_dilation, binary_erosion
from Skeleton_model.model import CustomUNet, transform
from Skeleton_model.Baseline_model import SkeletonBaselineModel
import torch
import time
from tqdm import tqdm
from Skeleton_model.Evaluate_utils import convert_prediction
from MiccaiModel.backbone.unet3d import UNet_ODT
from skimage.morphology import skeletonize


def model_for_iterations(actual_skeleton, model, skeleton, transform, device, iterations=1):
    accumulated_prediction = np.zeros_like(actual_skeleton)  # Initialize an empty array to sum predictions
    if (isinstance(model, SkeletonBaselineModel)):
        # print ("Using Baseline model")
        iterations = 1
    elif (isinstance(model, UNet_ODT)):
        # print ("Using UNet_ODT model")
        iterations = 1

    # elif (isinstance(model, CustomUNet)):
    #     print ("Using CustomUNet model")

    for _ in range(iterations):
        #print ("Iteration: ", _)
        if (isinstance(model, SkeletonBaselineModel)):
            predicted_hole = model.get_prediction(actual_skeleton)[0]

        elif (isinstance(model, UNet_ODT)):
            # ODT takes in skeleton as input
            # print ("actual_skeleton.shape: ",actual_skeleton.shape)
            if not skeleton: # Make it skeleton
                actual_skeleton = skeletonize(actual_skeleton).astype(actual_skeleton.dtype)
            
            actual_skeleton_tensor = transform(actual_skeleton[np.newaxis, ...]).unsqueeze(0).to(device)  # Move to device
            # print ("actual_skeleton_tensor.shape: ",actual_skeleton_tensor.shape)
            with torch.no_grad():
                predicted_hole = model(actual_skeleton_tensor)
            predicted_hole = predicted_hole.cpu().squeeze().numpy()
            
            # print ("predicted_hole shape: ",predicted_hole.shape)

        elif (isinstance(model, CustomUNet)):
            # Apply the correct transform
            # print ("actual_skeleton.shape: ",actual_skeleton.shape)

            actual_skeleton_tensor = transform(actual_skeleton[np.newaxis, ...]).unsqueeze(0).to(device)  # Move to device

            # print ("actual_skeleton_tensor.shape: ",actual_skeleton_tensor.shape)
            # Predict the gaps
            with torch.no_grad():
                predicted_logits = model(actual_skeleton_tensor)
                predicted_hole = torch.sigmoid(predicted_logits)  # <- add sigmoid

            predicted_hole = predicted_hole.cpu().squeeze().numpy()

            # print actual skeleton and predicted hole sums
            # print(f"actual_skeleton.sum(): {actual_skeleton.sum()}, predicted_hole.sum(): {predicted_hole.sum()}")

        # print number of unqies in predicted_hole by printing how many is between 0 and 0.05, 0.05 and 0.1, 0.1 and 0.15 etc
        bins = np.arange(0, 0.1, 0.005)
        hist, bin_edges = np.histogram(predicted_hole, bins=bins)

        bin_counts = [f"[{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}): {hist[i]}" for i in range(len(hist))]
        # print("Predicted hole value counts per bin:", ", ".join(bin_counts))
        
        # Convert prediction
        predicted_hole = convert_prediction(predicted_hole)
        # sum of predicted_hole
        # print(f"predicted_hole.sum(): {predicted_hole.sum()}")
        
        if (skeleton and isinstance(model, UNet_ODT)):
            if skeleton:
                predicted_hole = skeletonize(predicted_hole).astype(predicted_hole.dtype)
            predicted_hole = predicted_hole - actual_skeleton

        # Accumulate predictions
        accumulated_prediction += predicted_hole  # Add to accumulated predictions

        # Update actual_skeleton with the predicted_hole for the next iteration
        actual_skeleton = np.maximum(actual_skeleton, predicted_hole)

    return convert_prediction(accumulated_prediction)  # Return the summed predictions