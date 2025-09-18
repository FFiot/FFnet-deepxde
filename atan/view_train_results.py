import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
import re

def load_and_view_results(results_path):
    """Load and visualize training results"""
    # Load results
    results = np.load(results_path, allow_pickle=True).item()
    x = results["x"]
    y = results["y"]
    p = results["p"]
    
    print(f"Loaded results from {results_path}")
    print(f"Input shape: {x.shape}, Ground truth shape: {y.shape}, Prediction shape: {p.shape}")
    print(f"Input range: a=[{x[:, 0].min():.3f}, {x[:, 0].max():.3f}], b=[{x[:, 1].min():.3f}, {x[:, 1].max():.3f}]")
    print(f"Ground truth range: [{y[:, 0].min():.3f}, {y[:, 0].max():.3f}]")
    print(f"Prediction range: [{p[:, 0].min():.3f}, {p[:, 0].max():.3f}]")
    
    # Create 3D visualization comparing y and p
    fig = plt.figure(figsize=(15, 6))
    
    # Ground truth (y)
    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = ax1.scatter(x[:, 0], x[:, 1], y[:, 0], 
                          c=y[:, 0], cmap='viridis', s=1, alpha=0.6)
    ax1.set_xlabel('a')
    ax1.set_ylabel('b')
    ax1.set_zlabel('angle (normalized)')
    ax1.set_title('Ground Truth (y)')
    ax1.set_zlim(-1, 1)
    plt.colorbar(scatter1, ax=ax1, shrink=0.5, aspect=20, label='angle (normalized)')
    
    # Prediction (p)
    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = ax2.scatter(x[:, 0], x[:, 1], p[:, 0], 
                          c=p[:, 0], cmap='viridis', s=1, alpha=0.6)
    ax2.set_xlabel('a')
    ax2.set_ylabel('b')
    ax2.set_zlabel('angle (normalized)')
    ax2.set_title('Prediction (p)')
    ax2.set_zlim(-1, 1)
    plt.colorbar(scatter2, ax=ax2, shrink=0.5, aspect=20, label='angle (normalized)')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and print error statistics
    error = np.abs(y - p)
    mae = np.mean(error)
    max_error = np.max(error)
    print(f"\nError Statistics:")
    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"Max Absolute Error: {max_error:.6f}")
    print(f"Error range: [{np.min(error):.6f}, {max_error:.6f}]")
    
    # Show some sample comparisons
    print(f"\nSample comparisons (first 10):")
    for i in range(min(10, len(x))):
        a, b = x[i, 0], x[i, 1]
        y_val, p_val = y[i, 0], p[i, 0]
        err = abs(y_val - p_val)
        print(f"a={a:.3f}, b={b:.3f} -> y={y_val:.3f}, p={p_val:.3f}, error={err:.3f}")

def find_epoch_files(train_results_dir="train_results"):
    """Find all epoch npy files in train_results directory"""
    if not os.path.exists(train_results_dir):
        print(f"Directory {train_results_dir} does not exist!")
        return []
    
    # Find all npy files
    pattern = os.path.join(train_results_dir, "*.npy")
    files = glob.glob(pattern)
    
    # Extract epoch numbers and sort
    epoch_files = []
    for file in files:
        filename = os.path.basename(file)
        # Extract epoch number from filename like "epoch_0010.npy"
        match = re.search(r'epoch_(\d+)\.npy', filename)
        if match:
            epoch_num = int(match.group(1))
            epoch_files.append((epoch_num, file))
    
    # Sort by epoch number
    epoch_files.sort(key=lambda x: x[0])
    return epoch_files

def view_all_epochs(train_results_dir="train_results"):
    """View all epoch results in sequence"""
    epoch_files = find_epoch_files(train_results_dir)
    
    if not epoch_files:
        print(f"No epoch files found in {train_results_dir}")
        return
    
    print(f"Found {len(epoch_files)} epoch files:")
    for epoch_num, file_path in epoch_files:
        print(f"  Epoch {epoch_num:04d}: {file_path}")
    
    print("\nPress Enter to view each epoch, or 'q' to quit...")
    
    for epoch_num, file_path in epoch_files:
        user_input = input(f"\nView Epoch {epoch_num:04d}? (Enter/q): ").strip().lower()
        if user_input == 'q':
            break
        
        print(f"\n{'='*60}")
        print(f"VIEWING EPOCH {epoch_num:04d}")
        print(f"{'='*60}")
        
        try:
            load_and_view_results(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
        
        print(f"\nCompleted viewing Epoch {epoch_num:04d}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # If specific file provided, view that file
        results_path = sys.argv[1]
        load_and_view_results(results_path)
    else:
        # Default: view all epochs in train_results directory
        view_all_epochs()
