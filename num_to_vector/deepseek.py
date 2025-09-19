import numpy as np
from sklearn.linear_model import LinearRegression

class num_to_vector():
    """
    Number to Vector Encoder/Decoder using Fourier Features
    Improved decoding method for better reconstruction at boundaries
    """
    
    def __init__(self, ff_num: int = 8, ff_stride: int = 1, ff_scale: float = 1.0, ff_wrap: float = None) -> None:
        self.ff_num = ff_num
        self.ff_stride = ff_stride
        self.ff_scale = ff_scale
        self.ff_wrap = ff_wrap
        
        # Generate quantile positions
        self.q = np.linspace(0.0, 1.0, 2 * self.ff_num + 1, dtype=np.float32)[self.ff_stride::self.ff_stride + 1]

    def encode(self, x):
        x = x.flatten()
        
        # Compute quantile values for the input data
        p = np.quantile(x, self.q)
        
        # Compute distances from each input value to each quantile position
        d = np.abs(x[:, np.newaxis] - p[np.newaxis, :])
        
        # Apply wrapping for periodic data if specified
        if self.ff_wrap is not None:
            d = np.minimum(d, self.ff_wrap - d)
        
        # Apply exponential weighting with scaling
        d = np.square(d)
        # return d, p

        d = d * self.ff_scale
        s = np.exp(-d)
        
        # Normalize to create probability-like distribution
        s = s / np.sum(s, axis=-1, keepdims=True)
        
        return s, p

if __name__ == "__main__":
    # Demo code for improved num_to_vector class
    x = np.linspace(0.0, 1.0, 361, dtype=np.float32)
    x = np.expand_dims(x, -1)
    
    y = x * 2 * np.pi
    y = np.sin(y)

    n2v = num_to_vector(ff_num=2, ff_stride=1, ff_scale=10.0, ff_wrap=1.0)
    
    v, p = n2v.encode(x)
    # Plot the curves with x as horizontal axis and vectors in v as vertical axis, divided into 3 subplots
    import matplotlib.pyplot as plt
    
    # Create figure with subplots based on vector dimension plus one for the sum
    num_components = v.shape[-1]
    fig, axes = plt.subplots(num_components + 1, 1, figsize=(10, 6))
    
    # Plot each component of the vector v
    for i in range(num_components):
        axes[i].plot(x.flatten(), v[:, i])
        axes[i].set_title(f'Vector Component {i+1}')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel(f'v[{i}]')
        axes[num_components].set_ylim(0, 2)  # Set y-axis range to 0-2
        axes[i].grid(True)
    
    # Plot the sum of all vector components
    vector_sum = np.sum(v, axis=-1)
    print(vector_sum)
    axes[num_components].plot(x.flatten(), vector_sum)
    axes[num_components].set_title('Vector Sum (Sum of All Components)')
    axes[num_components].set_xlabel('x')
    axes[num_components].set_ylabel('Sum(v)')
    axes[num_components].set_ylim(0, 2)  # Set y-axis range to 0-2
    axes[num_components].grid(True)
    
    plt.tight_layout()
    plt.show()

