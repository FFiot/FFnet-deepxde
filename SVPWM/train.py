import os
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

class trainer(FFtrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_dataloader(self, **kwargs):
        # Load SVPWM dataset
        dataset_path = kwargs.get('dataset_path', 'SVPWM/dataset_arg100.npy')
        data = np.load(dataset_path, allow_pickle=True).item()
        
        self.X = data["X"].astype(np.float32)  # [d, q, theta] range [-1,1]
        self.Y1 = data["Y1"].astype(np.float32)  # [T0, T1, T2] range [0,1] - 辅助信息
        self.Y2 = data["Y2"].astype(np.float32)  # [u, v, w] range [0,1] - 训练目标
        
        # 只训练 uvw (Y2)，输入是 X，使用原始数据集
        data_train = torch.tensor(self.X, dtype=torch.float32)
        data_test = torch.tensor(self.X, dtype=torch.float32)  # 验证也用原始数据
        target_train = torch.tensor(self.Y2, dtype=torch.float32)
        target_test = torch.tensor(self.Y2, dtype=torch.float32)  # 验证也用原始数据

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

        # Build layer size: input 3D -> hidden -> output 3D (只训练 uvw)
        layer_size = [3] + [hidden_widths] * hidden_layers + [3]
        if ff_num is None:
            net = FFlinear(layer_size, activations, "Glorot uniform")
        else:
            net = FFlinear(layer_size, activations, "Glorot uniform", ff_num=ff_num, init_data=self.X)

        # Print network summary
        try:
            summary(net, input_size=(3,), device='cpu')
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
        rmse = rse.mean()
        self.add_epoch_log('Meter/RMSE/val', rmse, epoch)

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
    parser.add_argument('--dataset_name', type=str, default='SVPWM-UVW', help='Name of the dataset')
    parser.add_argument('--hidden_layers', type=int, default=1, help='Number of hidden layers (>=0, 0 or None means no hidden layers)')
    parser.add_argument('--hidden_widths', type=int, default=16, help='Width of hidden layers (must be >= 1)')
    parser.add_argument('--activations', type=str, default='relu', choices=['relu', 'tanh', 'leakyrelu', 'swish'], help='Activation functions')
    
    parser.add_argument('--ff_num', type=int, default=12, help='Number of FF layers (None for pure FC)')
    parser.add_argument('--ff_radius', type=int, default=1, help='FF radius parameter (must be >= 0 or None)')
    parser.add_argument('--ff_intensity', type=float, default=0.8, help='FF intensity parameter (greater than 1/ff_num and less than or equal to 0.95)')

    parser.add_argument('--train_epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for training and validation')

    parser.add_argument('--learn_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warmup epochs')
    parser.add_argument('--scheduler', type=str, default='cos', choices=['none', 'cos', 'step'], help='Learning rate scheduler type (cos or step)')
    
    parser.add_argument('--log_dir', type=str, default='./runs', help='Directory to store tensorboard logs')
    parser.add_argument('--refuse_dir', type=str, default=None, help='Directory to continue training from existing logs (test directory will not be deleted)')

    args = parser.parse_args()
    args_dict = vars(args)
    print(json.dumps(args_dict, indent=4))

    trainer = trainer(**args_dict)
    trainer.start()