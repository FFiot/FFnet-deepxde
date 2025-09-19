import numpy as np

class num_to_vector():
    """
    Number to Vector Encoder/Decoder using Fourier Features
    
    This class implements a novel encoding scheme that converts scalar values to vector representations
    using quantile-based Fourier features. The algorithm creates a sparse vector representation
    where each element corresponds to a quantile position, and the values are computed using
    exponential distance weighting.
    
    Algorithm:
    1. Define quantile positions based on ff_num and ff_stride
    2. For each input value, compute distances to all quantile positions
    3. Apply exponential weighting with optional wrapping for periodic data
    4. Normalize to create probability-like vector representation
    """
    
    def __init__(self, ff_num: int = 8, ff_stride: int = 1, ff_scale: float = 1.0, ff_wrap: float = None) -> None:
        """
        Initialize the number to vector encoder
        
        Args:
            ff_num: Number of Fourier features (quantile positions)
            ff_stride: Stride for quantile selection
            ff_scale: Scaling factor for distance computation
            ff_wrap: Wrapping distance for periodic data (None for non-periodic)
        """
        self.ff_num = ff_num
        self.ff_stride = ff_stride
        self.ff_scale = ff_scale
        self.ff_wrap = ff_wrap
        
        # Generate quantile positions: create 2*ff_num+1 positions, then select with stride
        # This creates a sparse set of quantile positions for encoding
        self.q = np.linspace(0.0, 1.0, 2 * self.ff_num + 1, dtype=np.float32)[self.ff_stride::self.ff_stride + 1]

    def encode(self, x):
        """
        Encode scalar values to vector representation
        
        Args:
            x: Input array of scalar values
            
        Returns:
            s: Encoded vector representation (normalized exponential weights)
            p: Quantile values used for encoding
        """
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
        d = d * self.ff_scale
        s = np.exp(-d)  # Negative exponential for distance weighting
        
        # Normalize to create probability-like distribution
        s = s / np.sum(s, axis=-1, keepdims=True)
        
        return s, p

    def decode(self, s, p):
        """
        Decode vector representation back to scalar values
        
        Args:
            s: Encoded vector representation
            p: Quantile values used in encoding
            
        Returns:
            x: Decoded scalar values (inverse of exponential distance weighting)
        """
        # Decode by inverting the exponential distance weighting
        # s = exp(-d^2 * scale) / sum(exp(-d^2 * scale))
        # We need to find x such that the distances d = |x - p| produce the given s
        
        # Use weighted average as initial guess
        o = np.sum(s * p, axis=-1, keepdims=True)

        return o 

if __name__ == "__main__":
    # Demo code for num_to_vector class
    x = np.linspace(0.0, 1.0, 11, dtype=np.float32)
    x = np.expand_dims(x, -1)
    n2v = num_to_vector(ff_num=3, ff_stride=1, ff_scale=2.0, ff_wrap=2.0)

    s, p = n2v.encode(x)
    print("Quantile values (p):")
    print(np.round(p, 3))

    x_decoded = n2v.decode(s, p)
        
    # Print each row with original value, decoded value, error, and encoded vector
    print("\nRow-wise comparison:")
    print("Index | Original | Decoded  | Error  | Encoded Vector")
    print("-" * 80)
    for i in range(len(x)):
        original = float(np.round(x[i, 0], 3))
        decoded = float(np.round(x_decoded[i, 0], 3))
        error = float(np.round(np.abs(x[i, 0] - x_decoded[i, 0]), 3))
        encoded = np.round(s[i], 3)
        print(f"{i:5d} | {original:8.3f} | {decoded:8.3f} | {error:6.3f} | {encoded}")
    
    print(f"\nMean reconstruction error: {np.round(np.mean(np.abs(x.flatten() - x_decoded)), 3)}")
