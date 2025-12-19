import os
import torch
import pprint
from tqdm import tqdm
from omegaconf import OmegaConf
from accelerate.state import AcceleratorState
from util.misc import flatten_dict
from util.accelerator import get_accelerator


class Trainer:
    def __init__(self, config):
        self.config = config
        self.accelerator, self.output_dir = get_accelerator(config)
        self.config.device_count = self.accelerator.num_processes
        
        self.device = self.accelerator.device
        if self.accelerator.mixed_precision == "bf16":
            self.dtype = torch.bfloat16
        elif self.accelerator.mixed_precision == "fp16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        self.global_step = self.config.train.global_step if self.config.train.global_step is not None else 0
        self.epoch = 0
        self.progress_bar = tqdm(
            total   = self.config.train.num_iter,
            initial = self.global_step,
            desc    = "Steps",
            disable = not self.accelerator.is_local_main_process,
        )

        self._load_models()
        self._load_optimizer()
        self._load_dataloader()

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(config.train.proj_name, config=flatten_dict(config))
            with open(os.path.join(self.output_dir, "config.yaml"), "w") as f:
                OmegaConf.save(self.config, f)

        self.accelerator.print("=" * 80)
        self.accelerator.print("Configuration:")
        self.accelerator.print(pprint.pformat(OmegaConf.to_container(config, resolve=True), indent=2, width=120).strip('{}'))
        AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = config.data.batch_size
        self.accelerator.print(AcceleratorState().deepspeed_plugin.deepspeed_config)
        self.accelerator.print("=" * 80)
        self.accelerator.print(f"Learnable parameters: {sum(p.numel() for p in self.params_to_learn if p.requires_grad) / 1e6} M")
        self.accelerator.print(f"Accelerator mixed precision: {self.accelerator.mixed_precision}")
        self.accelerator.print("=" * 80)
        print(
            "rank:", self.accelerator.state.process_index,
            "local_rank:", self.accelerator.state.local_process_index,
            "world:", self.accelerator.state.num_processes,
        )
        print("=" * 80)

    def _load_models(self):
        raise NotImplementedError

    def _load_dataloader(self):
        raise NotImplementedError

    def _load_optimizer(self):
        from transformers import get_cosine_schedule_with_warmup
        
        self.params_to_learn = list(p for p in self.model.parameters() if p.requires_grad)
        self.optimizer = torch.optim.AdamW(
            self.params_to_learn,
            lr           = self.config.train.lr,
            betas        = (0.9, 0.95),
            weight_decay = 5e-2,
            eps          = 1e-8,
        )
        
        # Learning rate scheduler with warmup
        use_scheduler = getattr(self.config.train, 'use_scheduler', True)
        if use_scheduler:
            warmup_steps = getattr(self.config.train, 'warmup_steps', 1000)
            min_lr = getattr(self.config.train, 'min_lr', 0.0)
            
            # 如果设置了 min_lr，需要创建一个支持 min_lr 的 scheduler
            if min_lr > 0:
                # 使用 lambda scheduler 来实现带 min_lr 的 cosine schedule
                from torch.optim.lr_scheduler import LambdaLR
                import math
                
                def lr_lambda(current_step):
                    if current_step < warmup_steps:
                        # Warmup phase: linear increase from min_lr to lr
                        return min_lr / self.config.train.lr + (1 - min_lr / self.config.train.lr) * (current_step / warmup_steps)
                    else:
                        # Cosine decay phase: from lr to min_lr
                        progress = (current_step - warmup_steps) / (self.config.train.num_iter - warmup_steps)
                        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                        return min_lr / self.config.train.lr + (1 - min_lr / self.config.train.lr) * cosine_decay
                
                self.scheduler = LambdaLR(self.optimizer, lr_lambda)
            else:
                # 使用默认的 cosine schedule with warmup
                self.scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=self.config.train.num_iter,
                )
        else:
            # 不使用 scheduler，创建一个 dummy scheduler 保持学习率不变
            from torch.optim.lr_scheduler import LambdaLR
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: 1.0)

    def train(self):
        raise NotImplementedError