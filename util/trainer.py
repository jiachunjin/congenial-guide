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
        warmup_steps = getattr(self.config.train, 'warmup_steps', 1000)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.config.train.num_iter,
        )
        
        # Resume 时：让 scheduler 快进到正确的步数
        if self.global_step > 0:
            self.scheduler.last_epoch = self.global_step
            print(f"Scheduler resumed to step {self.global_step}, lr = {self.scheduler.get_last_lr()[0]:.2e}")

    def train(self):
        raise NotImplementedError