"""
AGENT USAGE DOCUMENTATION
========================

This module implements FFtrainer - a comprehensive PyTorch training framework with advanced features.
FFtrainer provides a flexible, callback-based training system with automatic mixed precision, 
gradient clipping, learning rate scheduling, checkpointing, and TensorBoard logging.

CORE CAPABILITY - FLEXIBLE TRAINING FRAMEWORK:
==============================================

FFtrainer's primary strength lies in its modular, callback-based architecture that allows 
complete customization of the training process while providing robust default implementations:

MODULAR COMPONENT SYSTEM:
FFtrainer separates training components into configurable modules:
- Data Loading: Custom dataloader creation with FFdataloader utility
- Model Architecture: Flexible network creation via create_net()
- Device Management: Automatic GPU/CPU detection and allocation
- Loss Functions: Customizable loss computation with callback hooks
- Optimizers: Configurable optimizer setup with parameter control
- Learning Rate Scheduling: Dual scheduler system (warmup + training)
- Logging: Comprehensive TensorBoard integration with custom metrics

CALLBACK-BASED CUSTOMIZATION:
The framework provides 12+ callback hooks for complete training control:
- Data Transformation: Pre-processing hooks for train/eval data
- Loss Computation: Custom loss calculation with epoch/step context
- Training Events: Start/end callbacks for training phases
- Epoch Management: Per-epoch callbacks for custom logic
- Step Processing: Per-batch callbacks for detailed control

ADVANCED TRAINING FEATURES:
- Automatic Mixed Precision (AMP) support for memory efficiency
- Gradient clipping with configurable max_norm
- Warmup learning rate scheduling
- Automatic checkpoint saving and resuming
- Comprehensive logging with TensorBoard integration
- Memory management with CUDA cache clearing

This flexible training framework makes FFtrainer ideal for:
- Research experiments requiring custom training logic
- Production training with robust error handling
- Educational purposes with clear separation of concerns
- Rapid prototyping with minimal boilerplate code

QUICK START FOR AGENTS:
======================

1. BASIC USAGE:
   ```python
   import torch
   from FFtrainer import FFtrainer
   
   # Create a custom trainer by inheriting from FFtrainer
   class MyTrainer(FFtrainer):
       def create_net(self, **kwargs):
           return torch.nn.Linear(10, 1)
       
       def create_dataloader_train(self, **kwargs):
           x = torch.randn(1000, 10)
           y = torch.randn(1000, 1)
           return FFdataloader(x, y, batch_size=32)
       
       def create_dataloader_val(self, **kwargs):
           x = torch.randn(200, 10)
           y = torch.randn(200, 1)
           return FFdataloader(x, y, batch_size=32, shuffle=False)
   
   # Initialize and start training
   trainer = MyTrainer(
       train_epochs=100,
       model_name='MyModel',
       dataset_name='MyDataset'
   )
   trainer.start()
   ```

2. ADVANCED CONFIGURATION:
   ```python
   # Custom configuration with all features
   trainer = MyTrainer(
       # Training parameters
       train_epochs=200,
       warmup_epochs=10,
       use_amp=True,              # Enable automatic mixed precision
       max_norm=1.0,              # Gradient clipping
       
       # Device configuration
       device='gpu',              # 'gpu', 'cpu', or 'auto'
       gpu_id=0,                  # GPU device ID
       
       # Logging configuration
       model_name='AdvancedModel',
       dataset_name='AdvancedDataset',
       log_dir='./experiments',   # Custom log directory
       refuse_dir=None,           # Resume from existing directory
       
       # Custom parameters
       learning_rate=0.001,
       weight_decay=1e-4
   )
   ```

3. CUSTOM CALLBACK IMPLEMENTATION:
   ```python
   class CustomTrainer(FFtrainer):
       def create_net(self, **kwargs):
           return torch.nn.Sequential(
               torch.nn.Linear(784, 128),
               torch.nn.ReLU(),
               torch.nn.Linear(128, 10)
           )
       
       def create_loss_function(self, **kwargs):
           return torch.nn.CrossEntropyLoss()
       
       def create_optimizer(self, **kwargs):
           return torch.optim.Adam(
               self.net.parameters(),
               lr=kwargs.get('learning_rate', 0.001),
               weight_decay=kwargs.get('weight_decay', 1e-4)
           )
       
       def create_scheduler_train(self, **kwargs):
           return torch.optim.lr_scheduler.StepLR(
               self.optimizer, step_size=30, gamma=0.1
           )
       
       # Custom data transformation
       def data_transform_callback_train(self, epoch, step, x, y):
           # Add noise for data augmentation
           if epoch < 50:
               x = x + torch.randn_like(x) * 0.1
           return x, y
       
       # Custom loss computation
       def loss_function_callback_train(self, epoch, step, x, y, p):
           base_loss = self.loss_fn(p, y)
           # Add regularization
           l2_reg = sum(p.pow(2.0).sum() for p in self.net.parameters())
           return base_loss + 0.01 * l2_reg
       
       # Custom evaluation metrics
       def epoch_end_callback_eval(self, epoch, x, y, p, loss):
           accuracy = self.top_k_acc(p, y, k=1)
           self.add_epoch_log('eval/accuracy', accuracy, epoch)
           self.add_epoch_log('eval/loss', loss.mean().item(), epoch)
   ```

KEY PARAMETERS FOR AGENTS:
==========================

TRAINING PARAMETERS:
- train_epochs: Total number of training epochs (default: 200)
- warmup_epochs: Number of warmup epochs for learning rate (default: 0)
- use_amp: Enable automatic mixed precision (default: False)
- max_norm: Gradient clipping threshold (default: None, no clipping)

DEVICE PARAMETERS:
- device: Device type - 'gpu', 'cpu', or 'auto' (default: 'auto')
- gpu_id: GPU device ID when using GPU (default: 0)

LOGGING PARAMETERS:
- model_name: Name for model identification (default: None)
- dataset_name: Name for dataset identification (default: None)
- log_dir: Base directory for logs (default: './runs')
- refuse_dir: Resume from existing directory (default: None)

CALLBACK METHODS (OVERRIDE THESE):
==================================

REQUIRED METHODS (Must implement):
- create_net(**kwargs): Create and return the neural network model
- create_dataloader_train(**kwargs): Create training dataloader
- create_dataloader_val(**kwargs): Create validation dataloader

OPTIONAL METHODS (Override as needed):
- create_loss_function(**kwargs): Create loss function (default: None)
- create_optimizer(**kwargs): Create optimizer (default: None)
- create_scheduler_train(**kwargs): Create training scheduler (default: None)

CALLBACK HOOKS (Override for custom behavior):
- train_start_callback(epoch): Called at training start
- epoch_start_callback(epoch): Called at each epoch start
- data_transform_callback_train(epoch, step, x, y): Transform training data
- data_transform_callback_eval(epoch, step, x, y): Transform evaluation data
- loss_function_callback_train(epoch, step, x, y, p): Custom training loss
- loss_function_callback_eval(epoch, step, x, y, p): Custom evaluation loss
- step_end_callback_train(epoch, step, x, y, p, loss): After each training step
- epoch_end_callback_train(epoch, x, y, p, loss): After training epoch
- epoch_end_callback_eval(epoch, x, y, p, loss): After evaluation epoch
- epoch_end_callback(epoch): After each epoch (train + eval)
- train_end_callback(epoch): Called at training end

UTILITY METHODS:
- add_epoch_log(key, val, epoch): Add epoch-level metric to TensorBoard
- add_step_log(key, val, step): Add step-level metric to TensorBoard
- save_checkpoint(epoch): Save model checkpoint
- load_checkpoint(file_path): Load model checkpoint
- top_k_acc(predict, y, k=1): Calculate top-k accuracy

FFdataloader UTILITY CLASS:
===========================

FFdataloader provides a simple, efficient dataloader for PyTorch tensors:

```python
# Basic usage
x = torch.randn(1000, 10)  # Input features
y = torch.randn(1000, 1)   # Target values
dataloader = FFdataloader(x, y, batch_size=32, shuffle=True)

# Move to device
dataloader.to(device)

# Iterate through batches
for batch_x, batch_y in dataloader:
    # Process batch
    pass
```

PARAMETERS:
- x: Input tensor [samples, features]
- y: Target tensor [samples, targets]
- batch_size: Batch size (default: 32)
- shuffle: Whether to shuffle data (default: True)

TRAINING WORKFLOW:
==================

1. INITIALIZATION:
   ```python
   trainer = MyTrainer(**config)
   ```

2. AUTOMATIC SETUP (in start() method):
   - Create dataloaders via create_dataloader()
   - Create network via create_net()
   - Setup device via create_device()
   - Create loss function via create_loss_function()
   - Create optimizer via create_optimizer()
   - Setup schedulers via create_scheduler_warmup() and create_scheduler_train()
   - Setup logging directory and TensorBoard writer
   - Load existing checkpoints if available

3. TRAINING LOOP:
   ```python
   trainer.start()  # Starts the complete training process
   ```

4. AUTOMATIC FEATURES:
   - Checkpoint saving after each epoch
   - Automatic resume from latest checkpoint
   - TensorBoard logging of all metrics
   - Memory management and cleanup
   - Keyboard interrupt handling

CHECKPOINT SYSTEM:
==================

FFtrainer automatically saves checkpoints with the following structure:
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': self.net.state_dict(),
    'optimizer_state_dict': self.optimizer.state_dict(),
    'scheduler_state_dict': self.scheduler.state_dict(),
    'warmup_state_dict': self.warmup.state_dict() if self.warmup else None,
}
```

Checkpoints are saved as: `{log_dir}/epoch_{epoch}.pth`
The system automatically resumes from the latest checkpoint on startup.

LOGGING SYSTEM:
===============

FFtrainer provides comprehensive logging through TensorBoard:

AUTOMATIC LOGGING:
- train/lr: Learning rate at each step
- train/loss: Training loss at each step
- Model configuration as text (first epoch only)

CUSTOM LOGGING:
```python
# In your callback methods
self.add_epoch_log('custom/metric', value, epoch)
self.add_step_log('custom/step_metric', value, step)
```

Logs are saved to: `{log_dir}/dataset_name/model_name/timestamp/`

ADVANCED FEATURES:
==================

1. AUTOMATIC MIXED PRECISION:
   ```python
   trainer = MyTrainer(use_amp=True)
   # Automatically uses torch.amp.autocast and GradScaler
   ```

2. GRADIENT CLIPPING:
   ```python
   trainer = MyTrainer(max_norm=1.0)
   # Automatically clips gradients using torch.nn.utils.clip_grad_norm_
   ```

3. LEARNING RATE WARMUP:
   ```python
   trainer = MyTrainer(warmup_epochs=10)
   # Automatically implements linear warmup for first 10 epochs
   ```

4. MEMORY MANAGEMENT:
   - Automatic CUDA cache clearing after each epoch
   - Efficient tensor operations with proper cleanup
   - Non-blocking data transfers to GPU

5. ERROR HANDLING:
   - Graceful handling of KeyboardInterrupt
   - Proper cleanup of TensorBoard writer
   - Robust checkpoint loading with error recovery

TROUBLESHOOTING FOR AGENTS:
===========================

COMMON ISSUES:
1. "Must implement create_net": Override create_net() method
2. "Must implement create_dataloader_train": Override create_dataloader_train() method
3. "Must implement create_dataloader_val": Override create_dataloader_val() method
4. "GPU requested but not available": Set device='cpu' or device='auto'
5. Memory issues: Reduce batch_size or enable use_amp=True

PERFORMANCE TIPS:
- Use use_amp=True for memory efficiency and faster training
- Set appropriate max_norm for gradient clipping
- Use warmup_epochs for stable training start
- Implement efficient data_transform_callback methods
- Use FFdataloader for simple tensor-based datasets
- Enable proper logging for monitoring training progress

BEST PRACTICES:
- Always implement the three required methods (create_net, create_dataloader_train, create_dataloader_val)
- Use callbacks for custom behavior instead of modifying core training loop
- Implement proper evaluation metrics in epoch_end_callback_eval
- Use meaningful model_name and dataset_name for organization
- Save custom parameters in kwargs for access in callback methods
- Use TensorBoard logging for comprehensive training monitoring

INTEGRATION EXAMPLES:
=====================

1. WITH FFnormal LAYER:
   ```python
   class FFnormalTrainer(FFtrainer):
       def create_net(self, **kwargs):
           return torch.nn.Sequential(
               FFnormal(in_channels=64, ff_num=8),
               torch.nn.Linear(64*8, 10)
           )
   ```

2. WITH CUSTOM DATASET:
   ```python
   class CustomDatasetTrainer(FFtrainer):
       def create_dataloader_train(self, **kwargs):
           dataset = MyCustomDataset(**kwargs)
           return torch.utils.data.DataLoader(
               dataset, batch_size=kwargs.get('batch_size', 32), shuffle=True
           )
   ```

3. WITH MULTIPLE LOSSES:
   ```python
   def loss_function_callback_train(self, epoch, step, x, y, p):
       loss1 = self.loss_fn(p, y)
       loss2 = self.auxiliary_loss(p, x)
       return loss1 + 0.1 * loss2
   ```

This comprehensive training framework provides the flexibility and robustness needed for 
both research and production machine learning applications.
"""

import os
import json
from tracemalloc import start
from tqdm import tqdm
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

class FFdataloader:
    def __init__(self, x: torch.Tensor, y: torch.Tensor, batch_size: int = 32, shuffle: bool = True) -> None:
        self.x = x
        self.y = y

        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return (len(self.x) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            idx = torch.randperm(len(self.x), device=self.x.device)
        else:
            idx = torch.arange(len(self.x), device=self.x.device)
        
        for start_idx in range(0, len(idx), self.batch_size):
            batch_idx = idx[start_idx:start_idx + self.batch_size]
            yield self.x[batch_idx], self.y[batch_idx]

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        

class FFtrainer(torch.nn.Module):
    def __init__(self, **kwargs):
        super(FFtrainer, self).__init__()
        self.kwargs = kwargs

    def create_dataloader(self, **kwargs):
        return self.create_dataloader_train(**kwargs), self.create_dataloader_val(**kwargs)

    def create_dataloader_train(self, **kwargs):
       return None

    def create_dataloader_val(self, **kwargs):
        return None

    def create_net(self, **kwargs):
        pass

    def create_device(self, **kwargs):
        device_type = kwargs.get('device', 'auto')
        gpu_id = kwargs.get('gpu_id', 0)
        
        if device_type == 'gpu':
            if torch.cuda.is_available():
                return torch.device(f'cuda:{gpu_id}')
            else:
                raise RuntimeError("GPU requested but not available")
        elif device_type == 'cpu':
            return torch.device('cpu')
        else:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_loss_function(self, **kwargs):
        return None

    def create_optimizer(self, **kwargs):
        return None
    
    def create_scheduler_warmup(self, **kwargs):
        warmup_epochs = kwargs.get('warmup_epochs', 0)
        warm_steps = warmup_epochs * len(self.dataloader_train)
        if warm_steps > 0:
            warmup = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: min(1.0, (step + 1) / warm_steps)
            )
        else:
            warmup = None
        return warmup

    def create_scheduler_train(self, **kwargs):
        return None

    def add_epoch_log(self, key, val, epoch):
        self.epoch_log = getattr(self, 'epoch_log', {})
        self.epoch_log.setdefault(epoch, {})[key] = val
        self.writer.add_scalar(key, val, epoch)

    def add_step_log(self, key, val, step):
        self.step_log = getattr(self, 'step_log', {})
        self.step_log.setdefault(step, {})[key] = val
        self.writer.add_scalar(key, val, step)

    def train_start_callback(self, epoch):
        pass
    
    def epoch_start_callback(self, epoch):
        pass

    def data_transform_callback_train(self, epoch, step, x, y):
        return x, y

    def loss_function_callback_train(self, epoch, step, x, y, p):
        return self.loss_fn(p, y)

    def step_end_callback_train(self, epoch, step, x, y, p, loss):
        LR = self.optimizer.param_groups[0]["lr"]
        self.add_step_log('train/lr', LR, step)
        loss = loss.mean().item()
        self.add_step_log('train/loss', loss, step)

    def epoch_end_callback_train(self, epoch, x, y, p, loss):
        return None

    def data_transform_callback_eval(self, epoch, step, x, y): 
        return x, y

    def loss_function_callback_eval(self, epoch, step, x, y, p):
        return self.loss_fn(p, y)
    
    def epoch_end_callback_eval(self, epoch, x, y, p, loss):
        return None

    def epoch_end_callback(self, epoch):
        pass

    def train_end_callback(self, epoch):
        pass

    def train_one_epoch(self, epoch, tqdm_disable=False):
        self.train()
        x_list, y_list, p_list, loss_list = [], [], [], []

        step = (epoch-1) * len(self.dataloader_train)
        for x, y in tqdm(self.dataloader_train, desc=f'Epoch {epoch}', unit=' Batch', ncols=0, disable=tqdm_disable, leave=False):
            step = step + 1

            x, y = self.data_transform_callback_train(epoch, step, x, y)
            if self.device is not None:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with torch.amp.autocast(device_type=self.device.type):
                    p = self(x)
                    loss = self.loss_function_callback_train(epoch, step, x, y, p)
            
                self.scaler.scale(loss.mean()).backward()

                if self.max_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=self.max_norm)
                
                current_scale = self.scaler.get_scale()

                self.scaler.step(self.optimizer)
                self.scaler.update()

                new_scale = self.scaler.get_scale()

                optimizer_stepped = new_scale >= current_scale
            else:
                p = self(x)
                loss = self.loss_function_callback_train(epoch, step, x, y, p)
                loss.mean().backward()

                if self.max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=self.max_norm)
                
                self.optimizer.step()
                optimizer_stepped = True

            loss_detach = loss.detach()
            p_detach = p.detach()
            del loss, p

            self.step_end_callback_train(epoch, step, x, y, p_detach, loss_detach)
            
            if optimizer_stepped and self.warmup is not None and epoch <= self.warmup_epochs:
                self.warmup.step()

            x_list.append(x.detach().cpu())
            y_list.append(y.detach().cpu())
            p_list.append(p_detach.detach().cpu())
            loss_list.append(loss_detach.detach().cpu())

        del x, y, p_detach, loss_detach
        if self.device is not None and self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        x = torch.cat(x_list) if x_list is not None else None
        y = torch.cat(y_list)
        p = torch.cat(p_list)
        loss = torch.cat(loss_list)
        
        self.epoch_end_callback_train(epoch, x, y, p, loss)

        if self.scheduler is not None and epoch > self.warmup_epochs:
            self.scheduler.step()
        
        return x, y, p, loss

    def eval_one_epoch(self, epoch, tqdm_disable=False):
        self.eval()
        x_list, y_list, p_list, loss_list = [], [], [], []

        step = (epoch-1) * len(self.dataloader_val)
        for x, y in tqdm(self.dataloader_val, desc=f'Epoch {epoch}', unit=' Batch', ncols=0, disable=tqdm_disable, leave=False):
            step = step + 1

            x, y = self.data_transform_callback_eval(epoch, step, x, y)
            
            if self.device is not None:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            
            with torch.inference_mode():
                if self.scaler is not None:
                    with torch.amp.autocast(device_type=self.device.type):
                        p = self(x)
                        loss = self.loss_function_callback_eval(epoch, step, x, y, p)
                else:
                    p = self(x)
                    loss = self.loss_function_callback_eval(epoch, step, x, y, p)

            x_list.append(x.detach().cpu())
            y_list.append(y.detach().cpu())
            p_list.append(p.detach().cpu())
            loss_list.append(loss.detach().cpu())
        
        del x, y, p, loss
        if self.device is not None and self.device.type == 'cuda':
            torch.cuda.empty_cache()

        x = torch.cat(x_list) if x_list is not None else None
        y = torch.cat(y_list)
        p = torch.cat(p_list)
        loss = torch.cat(loss_list)

        self.epoch_end_callback_eval(epoch, x, y, p, loss)
        
        return x, y, p, loss
    
    def forward(self, x):
        return self.net(x)

    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'warmup_state_dict': self.warmup.state_dict() if self.warmup is not None else None,
        }
        torch.save(checkpoint, os.path.join(self.log_dir, f'epoch_{epoch}.pth'))

    def load_checkpoint(self, file_path):
        checkpoint = torch.load(os.path.join(self.log_dir, file_path), weights_only=True)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.warmup is not None and checkpoint['warmup_state_dict'] is not None:
            self.warmup.load_state_dict(checkpoint['warmup_state_dict'])

    def start(self):
        self.dataloader_train, self.dataloader_val = self.create_dataloader(**self.kwargs)
        self.net = self.create_net(**self.kwargs)
        self.device = self.create_device(**self.kwargs)

        self.net.to(self.device)

        self.loss_fn = self.create_loss_function(**self.kwargs)
        self.optimizer = self.create_optimizer(**self.kwargs)
        self.warmup = self.create_scheduler_warmup(**self.kwargs)
        self.scheduler = self.create_scheduler_train(**self.kwargs)

        self.warmup_epochs = self.kwargs.get('warmup_epochs', 0)
        self.train_epochs = self.kwargs.get('train_epochs', 200)
        self.use_amp = self.kwargs.get('use_amp', False)
        self.max_norm = self.kwargs.get('max_norm', None)
        self.model_name = self.kwargs.get('model_name', None)
        self.dataset_name = self.kwargs.get('dataset_name', None)

        refuse_dir = self.kwargs.get('refuse_dir', None)
        if refuse_dir is not None:
            self.log_dir = refuse_dir
        else:
            self.log_dir = self.kwargs.get('log_dir', './runs')
            self.log_dir = os.path.join(self.log_dir, self.dataset_name, self.model_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
        
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.start_epoch = 1
        files_path = [f for f in os.listdir(self.log_dir) if f.endswith('.pth') and f.startswith('epoch_')]
        if files_path:
            files_path.sort(reverse=True, key=lambda f: int(f.split('_')[1].split('.')[0]))
            latest_file = files_path[0]     
            self.load_checkpoint(latest_file)

        self.writer = SummaryWriter(log_dir=self.log_dir)
        try:
            if self.start_epoch == 1:
                args_dict = self.kwargs if isinstance(self.kwargs, dict) else vars(self.kwargs)
                self.writer.add_text('Model Configuration', json.dumps(args_dict, indent=4), 0)
            try:
                self.train_start_callback(self.start_epoch)
                self.scaler = torch.amp.GradScaler() if self.use_amp else None
                for epoch in range(self.start_epoch, self.train_epochs+1):
                    self.epoch_start_callback(epoch)
                    x, y, p, loss = self.train_one_epoch(epoch)
                    x, y, p, loss = self.eval_one_epoch(epoch)
                    self.epoch_end_callback(epoch)
                self.train_end_callback(epoch)
            except KeyboardInterrupt:
                print("\nTraining interrupted by user. Closing writer...")
        finally:
            self.writer.close()
            print("Writer has been closed.")

    @staticmethod
    def top_k_acc(predict, y, k=1):
        topk_pred = torch.topk(predict, k, dim=1).indices
        correct = torch.any(topk_pred == y.view(-1, 1), dim=1)
        top_k_acc = torch.sum(correct).item() / y.size(0)
        return top_k_acc