"""
FFnet: Physics-Informed Neural Network with FFnormal layers
Combines DeepXDE's FNN with FFnormal's focus point mechanism for enhanced PINN performance
"""

import os
import sys

import torch
import numpy as np
import torch.nn as nn

import deepxde as dde
from deepxde.nn import FNN

from FFnormal.FFnormal import FFnormal

# Add FFnormal to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'FFnormal'))

class FFnet(nn.Module):
    def __init__(self, layer_size, activation, initializer, ff_num, ff_intensity=0.8, init_points=None, init_data=None, ff_radius=1):
        super(FFnet, self).__init__()
        self.activation = activation
        self.regularizer = None  # Add regularizer attribute for DeepXDE compatibility
        self._ff_has_init_data = []  # Track which channels used init_data
        
        if ff_num is None:
            # Use standard FNN if no focus points specified
            self.net = FNN(layer_size, activation, initializer)
        else:
            # Create independent FFnormal layers for each input channel
            if not isinstance(ff_num, list):
                ff_num = [ff_num] * layer_size[0]
            
            self.ff_normal_layers = nn.ModuleList()
            self.ff_fc_layers = nn.ModuleList()
            for i in range(layer_size[0]):
                layer_init_points = init_points[:, i:i+1] if init_points is not None else None
                layer_init_data = init_data[:, i:i+1] if init_data is not None else None
                self.ff_normal_layers.append(
                    FFnormal(
                        in_channels=1,
                        ff_num=ff_num[i],
                        ff_radius=ff_radius,
                        ff_intensity=ff_intensity,
                        init_points=layer_init_points,
                        init_data=layer_init_data,
                    )
                )
                # Add FC layer after each FFnormal for feature enhancement and compression
                self.ff_fc_layers.append(
                    nn.Linear(ff_num[i], 1)
                )
                self._ff_has_init_data.append(layer_init_data is not None)
            
            # Create remaining layers using FNN (includes activation functions and weight initialization)
            # Input size is now the number of input channels (each compressed to 1)
            layer_size[0] = sum(ff_num) # Keep original input channel count
            self.remaining_net = FNN(layer_size, activation, initializer)

    def forward(self, x):
        if hasattr(self, 'net'):
            return self.net(x)
        else:
            # Process each input channel independently
            ff_outputs = [self.ff_normal_layers[i](x[:, i:i+1]) for i in range(x.shape[1])]
            # Apply FC layer to each FFnormal output for feature enhancement and compression
            # fc_outputs = [self.ff_fc_layers[i](ff_outputs[i]) for i in range(x.shape[1])]
            # Concatenate compressed outputs
            y = torch.cat(ff_outputs, dim=1)
            return self.remaining_net(y)
    
    def print_init_points(self):
        """Print FFnormal points for channels initialized via init_data."""
        if not hasattr(self, 'ff_normal_layers'):
            print("No FFnormal layers present.")
            return
        for idx, (layer, used_init_data) in enumerate(zip(self.ff_normal_layers, self._ff_has_init_data)):
            if used_init_data:
                pts = layer.points.detach().cpu().numpy()
                print(f"FFnormal channel {idx} points shape={tuple(pts.shape)}")
                print(pts)
    
# Example usage
if __name__ == "__main__":
    # Create FFnet with FFnormal
    # Build dummy init_data matching input layout: [N, C]
    init_data = torch.randn(1000, 2)
    ffnet = FFnet(
        layer_size=[2, 20, 20, 20, 1],
        activation="tanh",
        initializer="Glorot normal",
        ff_num=4,
        ff_radius=1,
        ff_intensity=0.8,
        init_data=init_data
    )

    # Print initialized points from init_data
    ffnet.print_init_points()

    # Test forward pass
    x = torch.randn(100, 2)
    print("FFnet output shape:", ffnet(x).shape)