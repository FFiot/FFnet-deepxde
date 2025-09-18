import os
import json
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

    def epoch_end_callback_eval(self, epoch, x, y, p, loss):
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
        self.model_name = self.kwargs.get('model_name', 'FFnetV1')
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