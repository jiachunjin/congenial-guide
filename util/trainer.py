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
        self.accelerator.print(f"Learnable parameters: {sum(p.numel() for p in self.params_to_learn if p.requires_grad) / 1e6} M")
        self.accelerator.print(f"Accelerator mixed precision: {self.accelerator.mixed_precision}")
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
        
        # Learning rate scheduler with warmup, 使用 self.global_step 作为步数
        from torch.optim.lr_scheduler import LambdaLR
        import math
        
        use_scheduler = getattr(self.config.train, 'use_scheduler', True)
        if use_scheduler:
            warmup_steps = getattr(self.config.train, 'warmup_steps', 1000)
            min_lr = getattr(self.config.train, 'min_lr', 0.0)
            # warmup_start_lr: warmup 起始学习率，如果未指定则默认为 0
            # 可以通过配置 train.warmup_start_lr 来指定 warmup 的起始学习率
            warmup_start_lr = getattr(self.config.train, 'warmup_start_lr', 0.0)
            
            # 使用 lambda scheduler，基于 self.global_step 来计算学习率
            def lr_lambda(_):
                # 使用 self.global_step 而不是 scheduler 传入的 step
                current_step = self.global_step
                if current_step < warmup_steps:
                    # Warmup phase: linear increase from warmup_start_lr to lr
                    warmup_ratio = current_step / warmup_steps
                    if warmup_start_lr > 0:
                        return warmup_start_lr / self.config.train.lr + (1 - warmup_start_lr / self.config.train.lr) * warmup_ratio
                    else:
                        return warmup_ratio
                else:
                    # Cosine decay phase: from lr to min_lr
                    progress = (current_step - warmup_steps) / max(1, self.config.train.num_iter - warmup_steps)
                    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                    if min_lr > 0:
                        return min_lr / self.config.train.lr + (1 - min_lr / self.config.train.lr) * cosine_decay
                    else:
                        return cosine_decay
            
            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        else:
            # 不使用 scheduler，创建一个 dummy scheduler 保持学习率不变
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda _: 1.0)

    def train(self):
        raise NotImplementedError