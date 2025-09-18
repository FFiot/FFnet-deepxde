"""
AGENT USAGE DOCUMENTATION
========================

1. BASIC USAGE:
   ```python
   import torch
   from FFlinear import FFlinear
   
   # Create a basic FFlinear for 2D physics problems
   ffnet = FFlinear(
       layer_size=[2, 20, 20, 20, 1],  # [input, hidden1, hidden2, hidden3, output]
       activation="tanh",               # Activation function
       initializer="Glorot normal",     # Weight initialization
       ff_num=4,                       # Number of focus points per channel
       ff_radius=1,                    # Focus point radius
       ff_intensity=0.8                # Focus intensity
   )
   
   # Process physics data
   x = torch.randn(100, 2)  # [batch, spatial_dimensions]
   output = ffnet(x)        # Physics solution
   ```

2. ADVANCED CONFIGURATION:
   ```python
   # Custom configuration for complex physics problems
   ffnet = FFlinear(
       layer_size=[3, 50, 50, 50, 50, 1],  # Deeper network
       activation="tanh",
       initializer="Glorot normal",
       ff_num=[8, 6, 4],                   # Different focus points per channel
       ff_radius=2,                        # Larger influence radius
       ff_intensity=0.9,                   # Higher focus intensity
       init_data=physics_training_data     # Physics-aware initialization
   )
   ```

3. PHYSICS-AWARE INITIALIZATION:
   ```python
   # Initialize with domain-specific physics data
   physics_data = torch.randn(1000, 2)  # [samples, spatial_dimensions]
   
   ffnet = FFlinear(
       layer_size=[2, 30, 30, 1],
       activation="tanh",
       initializer="Glorot normal",
       ff_num=6,
       ff_radius=1,
       ff_intensity=0.8,
       init_data=physics_data  # Learn focus points from physics data
   )
   
   # Print learned focus points
   ffnet.print_init_points()
   ```

4. DEEPXDE INTEGRATION:
   ```python
   import deepxde as dde
   
   # Use FFlinear as the neural network in DeepXDE
   net = FFlinear(
       layer_size=[2, 50, 50, 1],
       activation="tanh",
       initializer="Glorot normal",
       ff_num=8,
       ff_radius=1
   )
   
   # Create DeepXDE model with FFlinear
   model = dde.Model(data, net)
   model.compile("adam", lr=0.001)
   model.train(epochs=10000)
   ```

KEY PARAMETERS FOR AGENTS:
==========================

REQUIRED PARAMETERS:
- layer_size: List defining network architecture [input, hidden1, ..., output]
- activation: Activation function ("tanh", "relu", "sigmoid", "leakyrelu", "swish")
- initializer: Weight initialization method ("Glorot uniform", "He uniform", etc.)
- ff_num: Number of focus points per input channel (int or list, None for pure FC)

ACTIVATION FUNCTIONS:
- "tanh": Hyperbolic tangent (sigmoid family) → pairs with Xavier initialization
- "relu": Rectified Linear Unit (ReLU family) → pairs with He initialization
- "leakyrelu": Leaky ReLU (ReLU family) → pairs with He initialization
- "sigmoid": Sigmoid function (sigmoid family) → pairs with Xavier initialization
- "swish": Swish activation (ReLU family) → pairs with He initialization

INITIALIZATION METHODS:
- "Glorot uniform" / "Glorot normal": Xavier initialization (optimal for tanh, sigmoid)
- "He uniform" / "He normal": Kaiming initialization (optimal for relu, leakyrelu, swish)
- Smart pairing: FFlinear automatically selects best initialization based on activation function

FF_ PARAMETERS (Focus Point Configuration):
- ff_num: Number of focus points per channel (int or list of ints)
- ff_radius: Local influence range (default: 1)
- ff_intensity: Focus strength (default: 0.8)
- init_points: Pre-defined focus points (2D array: [channels, ff_num])
- init_data: Data-driven initialization (Tensor: [samples, channels])

ARCHITECTURE PARAMETERS:
- layer_size: Network architecture specification
- activation: Activation function for hidden layers (auto-paired with optimal initializer)
- initializer: Weight initialization strategy (can be overridden by smart pairing)

INITIALIZATION OPTIONS:
=======================

1. DEFAULT INITIALIZATION (no init_data or init_points):
   ```python
   ffnet = FFlinear(
       layer_size=[2, 20, 20, 1],
       activation="tanh",
       initializer="Glorot normal",
       ff_num=4
   )
   # Focus points initialized using FFnormal defaults
   ```

2. DATA-DRIVEN INITIALIZATION (init_data):
   ```python
   # Learn focus points from physics training data
   training_data = torch.randn(1000, 2)  # [samples, spatial_dimensions]
   
   ffnet = FFlinear(
       layer_size=[2, 20, 20, 1],
       activation="tanh",
       initializer="Glorot normal",
       ff_num=6,
       init_data=training_data
   )
   # Focus points automatically learned from data distribution
   ```

3. PRE-DEFINED FOCUS POINTS (init_points):
   ```python
   import numpy as np
   
   # Define custom focus points for each channel
   custom_points = np.array([
       [-2.0, -1.0, 0.0, 1.0, 2.0],  # Channel 0 focus points
       [-1.5, -0.5, 0.5, 1.5, 2.5],  # Channel 1 focus points
   ])  # Shape: [channels, ff_num]
   
   ffnet = FFlinear(
       layer_size=[2, 20, 20, 1],
       activation="tanh",
       initializer="Glorot normal",
       ff_num=5,
       init_points=custom_points
   )
   ```

ACTIVATION FUNCTION AND INITIALIZATION PAIRING:
===============================================

FFlinear automatically pairs activation functions with appropriate initialization methods for optimal training:

ACTIVATION-INITIALIZER BEST PRACTICES:
- **ReLU Family** (relu, leakyrelu, swish) → **He Initialization** (Kaiming)
- **Sigmoid Family** (tanh, sigmoid) → **Xavier Initialization** (Glorot)
- **Smart Pairing**: FFlinear automatically selects the best initialization even if you specify a different one

SUPPORTED INITIALIZERS:
- "Glorot uniform" / "Glorot normal" (Xavier initialization)
- "He uniform" / "He normal" (Kaiming initialization)
- Any other name defaults to Xavier initialization

EXAMPLES:
```python
# ReLU with He initialization (automatic pairing)
ffnet = FFlinear([2, 50, 1], "relu", "He uniform", None)

# Tanh with Xavier initialization (automatic pairing)
ffnet = FFlinear([2, 50, 1], "tanh", "Glorot uniform", None)

# Smart pairing: ReLU with Glorot will automatically use He initialization
ffnet = FFlinear([2, 50, 1], "relu", "Glorot uniform", None)  # Uses He internally

# Smart pairing: Tanh with He will automatically use Xavier initialization
ffnet = FFlinear([2, 50, 1], "tanh", "He uniform", None)  # Uses Xavier internally
```

ARCHITECTURE CONFIGURATION:
===========================

1. STANDARD ARCHITECTURE:
   ```python
   # Basic physics-informed neural network with proper activation-initializer pairing
   ffnet = FFlinear(
       layer_size=[2, 50, 50, 1],  # 2D spatial input, 1D output
       activation="tanh",           # Sigmoid family
       initializer="Glorot normal", # Xavier initialization (optimal for tanh)
       ff_num=8
   )
   ```

2. DEEP ARCHITECTURE WITH RELU:
   ```python
   # Deeper network for complex physics using ReLU
   ffnet = FFlinear(
       layer_size=[3, 100, 100, 100, 100, 1],  # 3D spatial, deep network
       activation="relu",           # ReLU family
       initializer="He uniform",    # Kaiming initialization (optimal for ReLU)
       ff_num=12
   )
   ```

3. MULTI-OUTPUT ARCHITECTURE:
   ```python
   # Multiple output variables (e.g., velocity components)
   ffnet = FFlinear(
       layer_size=[2, 50, 50, 3],  # 2D input, 3D output
       activation="swish",          # ReLU family
       initializer="He normal",     # Kaiming initialization (optimal for Swish)
       ff_num=6
   )
   ```

4. CHANNEL-SPECIFIC FOCUS POINTS:
   ```python
   # Different focus points for different spatial dimensions
   ffnet = FFlinear(
       layer_size=[3, 50, 50, 1],
       activation="leakyrelu",      # ReLU family
       initializer="He uniform",    # Kaiming initialization (optimal for LeakyReLU)
       ff_num=[8, 6, 4],           # Different focus points per channel
       ff_radius=2
   )
   ```

DEEPXDE INTEGRATION:
====================

FFlinear is fully compatible with DeepXDE's physics-informed neural network framework:

1. BASIC DEEPXDE USAGE:
   ```python
   import deepxde as dde
   
   # Create FFlinear as the neural network
   net = FFlinear(
       layer_size=[2, 50, 50, 1],
       activation="tanh",
       initializer="Glorot normal",
       ff_num=8
   )
   
   # Use with DeepXDE
   model = dde.Model(data, net)
   model.compile("adam", lr=0.001)
   model.train(epochs=10000)
   ```

2. WITH CUSTOM TRAINING:
   ```python
   # Advanced training configuration
   net = FFlinear(
       layer_size=[2, 100, 100, 1],
       activation="tanh",
       initializer="Glorot normal",
       ff_num=12,
       init_data=physics_data
   )
   
   model = dde.Model(data, net)
   model.compile("adam", lr=0.001, loss_weights=[1, 1, 1])
   model.train(epochs=20000, display_every=1000)
   ```

3. MULTI-PHYSICS PROBLEMS:
   ```python
   # Complex multi-physics simulation
   net = FFlinear(
       layer_size=[3, 100, 100, 100, 2],  # 3D input, 2D output
       activation="tanh",
       initializer="Glorot normal",
       ff_num=[10, 8, 6],  # Channel-specific focus points
       ff_radius=2
   )
   ```

UTILITY METHODS:
================

1. FOCUS POINT INSPECTION:
   ```python
   # Print learned focus points
   ffnet.print_init_points()
   # Output: Shows focus points for channels initialized with init_data
   ```

2. FORWARD PASS:
   ```python
   # Standard forward pass
   x = torch.randn(100, 2)  # [batch, spatial_dimensions]
   output = ffnet(x)        # [batch, output_dimensions]
   ```

3. DEEPXDE COMPATIBILITY:
   ```python
   # FFlinear automatically provides DeepXDE compatibility attributes
   print(ffnet.activation)     # Activation function
   print(ffnet.regularizer)    # Regularizer (None by default)
   ```

TROUBLESHOOTING FOR AGENTS:
===========================

COMMON ISSUES:
1. "ff_num must be >= 4": Increase ff_num to at least 4
2. "layer_size must have at least 2 elements": Provide [input, ..., output] architecture
3. "init_data dimension error": Ensure init_data shape matches [samples, input_channels]
4. "init_points shape mismatch": Ensure init_points shape is [channels, ff_num]
5. Memory issues: Reduce ff_num or use smaller layer_size

PERFORMANCE TIPS:
- Use init_data for physics-aware initialization
- Choose appropriate ff_num based on problem complexity
- Use channel-specific ff_num for multi-dimensional problems
- Enable ff_radius > 0 for better local feature capture
- Use deeper networks (more hidden layers) for complex physics
- Initialize with domain-specific data for better convergence

BEST PRACTICES:
- Always provide meaningful layer_size architecture
- Use init_data when you have physics training data
- Choose activation functions appropriate for your physics problem
- Use appropriate weight initializers for your activation function
- Monitor focus point evolution during training
- Use print_init_points() to verify initialization

INTEGRATION EXAMPLES:
=====================

1. WITH FFtrainer:
   ```python
   from FFtrainer import FFtrainer
   
   class PhysicsTrainer(FFtrainer):
       def create_net(self, **kwargs):
           return FFlinear(
               layer_size=[2, 50, 50, 1],
               activation="tanh",
               initializer="Glorot normal",
               ff_num=8,
               init_data=kwargs.get('physics_data')
           )
   ```

2. WITH CUSTOM PHYSICS DATA:
   ```python
   # Load physics simulation data
   physics_data = load_physics_simulation_data()  # [samples, spatial_dims]
   
   ffnet = FFlinear(
       layer_size=[2, 50, 50, 1],
       activation="tanh",
       initializer="Glorot normal",
       ff_num=8,
       init_data=physics_data
   )
   ```

3. MULTI-SCALE PHYSICS:
   ```python
   # Different focus points for different scales
   ffnet = FFlinear(
       layer_size=[3, 100, 100, 1],
       activation="tanh",
       initializer="Glorot normal",
       ff_num=[12, 8, 4],  # Fine to coarse focus points
       ff_radius=2
   )
   ```
"""

import os
import sys

import torch
import numpy as np
import torch.nn as nn

from FFnormal.FFnormal import FFnormal

# Add FFnormal to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'FFnormal'))

class FFlinear(nn.Module):
    def __init__(self, layer_size, activation, initializer, ff_num=None, ff_intensity=0.8, init_points=None, init_data=None, ff_radius=1):
        super(FFlinear, self).__init__()
        self.activation = activation
        self.regularizer = None  # Add regularizer attribute for DeepXDE compatibility
        
        if ff_num is None or ff_num == 0:
            # Use standard fc if no focus points specified
            self.net = self._create_fc(layer_size, activation, initializer)
        else:
            # Create independent FFnormal layers for each input channel
            if not isinstance(ff_num, list):
                ff_num = [ff_num] * layer_size[0]
            
            self.ff_normal_layers = nn.ModuleList()
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
            
            # Input size is now the number of input channels (each compressed to 1)
            layer_size[0] = sum(ff_num) # Keep original input channel count
            self.remaining_net = self._create_fc(layer_size, activation, initializer)

    def _create_fc(self, layer_size, activation, initializer):
        """Create a fully connected neural network."""
        layers = []
        for i in range(len(layer_size) - 1):
            layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))
            if i < len(layer_size) - 2:  # Don't add activation after last layer
                if activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "leakyrelu":
                    layers.append(nn.LeakyReLU())
                elif activation == "sigmoid":
                    layers.append(nn.Sigmoid())
                elif activation == "swish":
                    layers.append(nn.SiLU())  # SiLU is the same as Swish
        
        network = nn.Sequential(*layers)
        
        # Initialize weights based on activation function and initializer
        network.apply(lambda m: self._init_weights(m, activation, initializer))
        
        return network
    
    def _init_weights(self, m, activation, initializer):
        """Initialize weights for linear layers based on activation function."""
        if isinstance(m, nn.Linear):
            if initializer == "Glorot uniform" or initializer == "Glorot normal":
                # Xavier/Glorot initialization - good for tanh, sigmoid
                if activation in ["tanh", "sigmoid"]:
                    nn.init.xavier_uniform_(m.weight)
                else:
                    # For ReLU family, use He initialization even with Glorot initializer
                    nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            elif initializer == "He uniform" or initializer == "He normal":
                # He initialization - good for ReLU, LeakyReLU, Swish
                if activation in ["relu", "leakyrelu", "swish"]:
                    nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                else:
                    # For tanh/sigmoid, use Xavier even with He initializer
                    nn.init.xavier_uniform_(m.weight)
            else:
                # Default to Xavier for unknown initializers
                nn.init.xavier_uniform_(m.weight)
            
            nn.init.zeros_(m.bias)

    def forward(self, x):
        if hasattr(self, 'net'):
            return self.net(x)
        else:
            # Process each input channel independently
            ff_outputs = [self.ff_normal_layers[i](x[:, i:i+1]) for i in range(x.shape[1])]
            # Concatenate outputs
            y = torch.cat(ff_outputs, dim=1)
            # y = torch.softmax(y, dim=-1)
            return self.remaining_net(y)
    
    def print_init_points(self):
        """Print FFnormal points for all channels."""
        if not hasattr(self, 'ff_normal_layers'):
            print("No FFnormal layers present.")
            return
        for idx, layer in enumerate(self.ff_normal_layers):
            pts = layer.points.detach().cpu().numpy()
            print(f"FFnormal channel {idx} points shape={tuple(pts.shape)}")
            print(pts)
    
# Example usage
if __name__ == "__main__":
    # Create FFlinear with FFnormal
    # Build dummy init_data matching input layout: [N, C]
    init_data = torch.randn(1000, 2)
    ffnet = FFlinear(
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
    print("FFlinear output shape:", ffnet(x).shape)