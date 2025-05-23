import torch
import torch.nn as nn
from monai.networks.nets import UNet

# Define your MONAI-based 3D U-Net
class CustomUNet(nn.Module):
    def __init__(self, channels=(32, 64, 128), strides=(2, 2), kernel_size=3):
        super().__init__()
        self.unet = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=channels,
            strides=strides,
            kernel_size=kernel_size,
            norm=None,
            act=None,
            num_res_units=0,
        )

    def forward(self, x):
        return self.unet(x)



def calc_rf(model, threshold=0.2, debug=False): 
    # Threshold for non-zero elements. Used because the output is not exactly 0 due to some noise from somewhere. Probably floating point errors.
    model.eval()

    # Set all Conv3d weights to 1 and biases to 0
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    # Create an input tensor with a single 1 in the center
    D, H, W = 256, 256, 256  # Ensure divisibility by 4 (2^depth)
    input_tensor = torch.zeros(1, 1, D, H, W)
    input_tensor[0, 0, D//2, H//2, W//2] = 1000.0


    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)

    if debug: print("Output min:", output.min().item())
    if debug: print("Output max:", output.max().item())

    
    # Thresholded mask
    nonzero_coords = (output.abs() > threshold).nonzero(as_tuple=False)


    if debug: 
        n = nonzero_coords.shape[0]
        cube_root = n ** (1/3)
        cube_root_rounded = round(cube_root)
        cubed_value = cube_root_rounded ** 3
        cubed_error = abs(n - cubed_value)
        print(f"Non-zero voxel count: {n}")
        print(f"Cube root ≈ {cube_root:.6f}")
        print(f"Rounded cube root: {cube_root_rounded}")
        print(f"Cube of rounded root: {cubed_value}")
        print(f"Error from cubed root: {cubed_error}")

    # Exclude batch and channel dimensions
    spatial_coords = nonzero_coords[:, 2:]  # shape: (N, 3) → D, H, W

    # Calculate the bounding box of the non-zero region
    min_coords = spatial_coords.min(0)[0]
    max_coords = spatial_coords.max(0)[0]
    rf_size = max_coords - min_coords + 1

    # Print the actual receptive field size (depth, height, width)
    if debug: print(f"Estimated receptive field size: {tuple(rf_size.tolist())} \n")
    return rf_size.tolist()[0]

# model = CustomUNet(channels=(32, 64, 128), strides=(2, 2), kernel_size=3)
# model = CustomUNet(channels=(32, 64, 128), strides=(2, 2), kernel_size=5)
# model = CustomUNet(channels=(32, 64), strides=(2,), kernel_size=3)

print ("Receptive Field Sizes for Different Configurations:")
print ("-------------------------------------------------")
print ("Deeper UNet with 3 levels:")
print(calc_rf(CustomUNet(channels=(32, 64, 128, 256), strides=(2,2,2), kernel_size=3)))
print ("Default")
print(calc_rf(CustomUNet(channels=(32, 64, 128), strides=(2,2), kernel_size=3)))
print ("Shallower UNet with 1 level:")
print(calc_rf(CustomUNet(channels=(32, 64), strides=(2,), kernel_size=3)))
print ("Default UNet, larger kernel:")
print(calc_rf(CustomUNet(channels=(32, 64, 128), strides=(2,2), kernel_size=5)))


