import os
from pickle import TRUE
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import datetime

import numpy as np

import torch
import torch.nn as nn
from torchsummary import summary

from FFlinear import FFlinear
from FFtrainer import FFtrainer, FFdataloader

def num_to_vec(x, ff_num, ff_stride, ff_scale=1.0, ff_wrap=None):
    qs = np.linspace(0.0, 1.0, 2 * ff_num + 1, dtype=np.float32)
    qs = qs[..., ff_stride::ff_stride + 1]

    x = np.asarray(x)
    if x.ndim == 1:
        xN = x.reshape(-1, 1)
    elif x.ndim == 2:
        xN = x
    else:
        raise ValueError("x must be 1D or 2D array")

    # per-dimension quantiles over samples axis (axis=0)
    # np.quantile returns shape (len(qs), D) for 2D input
    p = np.quantile(xN, qs, axis=0)
    # transpose to (D, K)
    p = p.T.astype(np.float32)

    # compute distances for each dimension then concatenate along features
    parts = []
    for d in range(xN.shape[1]):
        p_d = p[d]  # (K,)
        diff = np.abs(xN[:, d][:, None] - p_d[None, :])
        if ff_wrap is not None:
            diff = np.minimum(diff, ff_wrap - diff)
        diff = (diff ** 2) * ff_scale
        parts.append(np.exp(-diff))
    s = np.concatenate(parts, axis=1)
    s = s / np.sum(s, axis=1, keepdims=True)

    return s, p

class trainer(FFtrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_dataloader(self, **kwargs):
        # Load atan dataset
        dataset_path = kwargs.get('dataset_path', 'atan/dataset_arg100.npy')
        data = np.load(dataset_path, allow_pickle=True).item()
        
        self.X = data["X"].astype(np.float32)  # [a, b] range [-1,1]
        self.Angle = data["Y"].astype(np.float32)  # [angle] normalized to [-1,1]

        outdim = kwargs.get('outdim', 10)
        self.Y, self.Angle_point = num_to_vec(self.Angle, outdim, 1, 1.0, 2.0)

        p = self.Y * self.Angle_point
        p = np.sum(p, axis=-1, keepdims=-1)

        # 训练角度预测，输入是 X，使用原始数据集
        data_train = torch.tensor(self.X, dtype=torch.float32)
        data_test = torch.tensor(self.X, dtype=torch.float32)  # 验证也用原始数据
        target_train = torch.tensor(self.Y, dtype=torch.float32)
        target_test = torch.tensor(self.Y, dtype=torch.float32)  # 验证也用原始数据

        self.batch_size = self.kwargs.get('batch_size', 32)
        dataloader_train = FFdataloader(data_train, target_train, self.batch_size, True)
        dataloader_val = FFdataloader(data_test, target_test, len(data_test), False)
        return dataloader_train, dataloader_val

    def create_net(self, **kwargs):
        # Get model type and parameters
        hidden_layers = kwargs.get('hidden_layers', 3)
        hidden_widths = kwargs.get('hidden_widths', 64)
        activations = kwargs.get('activations', 'relu')
        
        ff_num=kwargs.get('ff_num', None)
        outdim = kwargs.get('outdim', 10)

        # Build layer size: input 2D -> hidden -> output 1D (角度预测)
        layer_size = [2] + [hidden_widths] * hidden_layers + [outdim]
        if ff_num is None:
            net = FFlinear(layer_size, activations, "Glorot uniform")
        else:
            net = FFlinear(layer_size, activations, "Glorot uniform", ff_num=ff_num, init_data=self.X)

        # Print network summary
        try:
            summary(net, input_size=(2,), device='cpu')
        except Exception as e:
            print(f"Warning: Could not print network summary: {e}")
            print(f"Network architecture: {layer_size}")

        net.print_init_points()

        return net

    def create_loss_function(self, **kwargs):
        return torch.nn.MSELoss(reduction='none')

    def create_optimizer(self, **kwargs):
        learn_rate = kwargs.get('learn_rate', 0.001)
        return torch.optim.Adam(self.parameters(), lr=learn_rate, weight_decay=1e-4)
        
    def create_scheduler_train(self, **kwargs):
        scheduler_type = kwargs.get('scheduler', 'none')
        if scheduler_type == 'cos':
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=kwargs.get('train_epochs', 200) - kwargs.get('warmup_epochs', 10))
        elif scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        else:
            return None
    
    def epoch_end_callback_train(self, epoch, x, y, p, loss):
        loss = loss.mean().item()
        self.add_epoch_log('Loss/train', round(loss, 8), epoch)

        rse = torch.square(y - p) / (torch.square(y) + 1e-3)
        rmse = rse.mean().item()
        self.add_epoch_log('Meter/RMSE/train', rmse, epoch)

    def epoch_end_callback_eval(self, epoch, x, y, p, loss):
        x, y, p = x.cpu().numpy(), y.cpu().numpy(), p.cpu().numpy()
        loss = loss.cpu().numpy()

        loss = np.mean(loss)
        self.add_epoch_log('Loss/val', round(loss, 8), epoch)

        rse = np.square(y - p) / (np.square(y) + 1e-3)
        rse = np.sum(rse, axis=-1)
        rmse = rse.mean()
        self.add_epoch_log('Meter/RMSE/val', rmse, epoch)
        
        if epoch % 10 == 0:
            # Save x, y, p to npy file with epoch number
            results = {
                "x": x.astype(np.float32),  # input [a, b]
                "y": y.astype(np.float32),  # ground truth angle (normalized)
                "p": p.astype(np.float32)   # predicted angle (normalized)
            }
            filename = f"atan/train_results/epoch_{epoch:04d}.npy"
            np.save(filename, results)

    def epoch_end_callback(self, epoch):
        # Get the latest metrics from epoch_log
        if hasattr(self, 'epoch_log') and epoch in self.epoch_log:
            epoch_data = self.epoch_log[epoch]
            
            # Extract metrics
            train_loss = epoch_data.get('Loss/train', 0.0)
            train_rmse = epoch_data.get('Meter/RMSE/train', 0.0)
            val_loss = epoch_data.get('Loss/val', 0.0)
            val_rmse = epoch_data.get('Meter/RMSE/val', 0.0)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print every 100 epochs or on the last epoch
            total_epochs = self.kwargs.get('train_epochs', 200)
            if epoch % 10 == 0 or epoch == total_epochs:
                print(f"Epoch {epoch:3d} | "
                      f"Train Loss: {train_loss:.6f} | Train RMSE: {train_rmse:.6f} | "
                      f"Val Loss: {val_loss:.6f} | Val RMSE: {val_rmse:.6f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Time: {datetime.datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FFPINN Configuration')

    parser.add_argument('--device', type=str, default='gpu', choices=['auto', 'cpu', 'gpu'], help='Device to use (auto, cpu, or gpu)')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU index to use when device is gpu')
    parser.add_argument('--use_amp', type=bool, default=True, help='Enable Automatic Mixed Precision training')
    
    parser.add_argument('--model_name', type=str, default='FFlinear', help='Name of the model')
    parser.add_argument('--dataset_name', type=str, default='atan-angle', help='Name of the dataset')
    parser.add_argument('--hidden_layers', type=int, default=2, help='Number of hidden layers (>=0, 0 or None means no hidden layers)')
    parser.add_argument('--hidden_widths', type=int, default=32, help='Width of hidden layers (must be >= 1)')
    parser.add_argument('--outdim', type=int, default=2, help='Output dimension of the model')
    parser.add_argument('--activations', type=str, default='relu', choices=['relu', 'tanh', 'leakyrelu', 'swish'], help='Activation functions')
    
    parser.add_argument('--ff_num', type=int, default=16, help='Number of FF layers (None for pure FC)')
    parser.add_argument('--ff_radius', type=int, default=0, help='FF radius parameter (must be >= 0 or None)')
    parser.add_argument('--ff_intensity', type=float, default=0.8, help='FF intensity parameter (greater than 1/ff_num and less than or equal to 0.95)')

    parser.add_argument('--train_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training and validation')

    parser.add_argument('--learn_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--warmup_epochs', type=int, default=1, help='Number of warmup epochs')
    parser.add_argument('--scheduler', type=str, default='cos', choices=['none', 'cos', 'step'], help='Learning rate scheduler type (cos or step)')
    
    parser.add_argument('--log_dir', type=str, default='./runs', help='Directory to store tensorboard logs')
    parser.add_argument('--refuse_dir', type=str, default=None, help='Directory to continue training from existing logs (test directory will not be deleted)')

    args = parser.parse_args()
    args_dict = vars(args)
    print(json.dumps(args_dict, indent=4))

    trainer = trainer(**args_dict)
    trainer.start()

