import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
from util.trainer import Trainer
from runner.mcq_gen.dev_ar_head import load_quantizer

def modify_internvl(internvl, config):
    visual_embeddings = nn.Embedding(config.quantizer.num_embeddings * config.quantizer.num_codebooks, config.llm_hidden_size)
    visual_embeddings.requires_grad_(True)
    num_params = sum(p.numel() for p in visual_embeddings.parameters() if p.requires_grad)
    print(f"Trainable parameters in visual_embeddings: {num_params / 1e6:.2f}M")

    head = nn.Linear(config.llm_hidden_size, config.quantizer.num_embeddings * config.quantizer.num_codebooks)
    head.requires_grad_(True)
    num_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"Trainable parameters in head: {num_params / 1e6:.2f}M")

    internvl.visual_embeddings = visual_embeddings
    internvl.head = head

    return internvl


class MyTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

    def _load_models(self):
        from transformers import AutoTokenizer
        from model.internvl.modeling_internvl_chat import InternVLChatModel

        quantizer = load_quantizer(self.config.model.quantizer)
        internvl = InternVLChatModel.from_pretrained(self.config.model.internvl_path)
        internvl = modify_internvl(internvl, self.config.model)

        if self.config.train.resume_path is not None:
            ckpt = torch.load(self.config.train.resume_path, map_location="cpu", weights_only=True)
            internvl.load_state_dict(ckpt, strict=True)
            print(f"internvl loaded from {self.config.train.resume_path}")

        tokenizer = AutoTokenizer.from_pretrained(self.config.model.internvl_path, trust_remote_code=True, use_fast=False)

        self.model = internvl
        self.quantizer = quantizer.to(self.device, self.dtype).eval()
        self.tokenizer = tokenizer

    def _load_dataloader(self):
        from util.dataloader import get_blip3o_dataloader
        self.dataloader = get_blip3o_dataloader(self.config.data, self.tokenizer, self.accelerator)

    def train(self):
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        imagenet_mean = torch.tensor(IMAGENET_MEAN, device=self.accelerator.device, dtype=self.dtype).view(1, 3, 1, 1)
        imagenet_std = torch.tensor(IMAGENET_STD, device=self.accelerator.device, dtype=self.dtype).view(1, 3, 1, 1)
        training_done = False

        while not training_done:
            for batch in self.dataloader:
                with self.accelerator.accumulate(self.model):
                    self.model.train()

                    pixel_values = batch["pixel_values"].to(self.device, self.dtype)
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)

                    x_gen = (pixel_values - imagenet_mean) / imagenet_std
                    B = x_gen.shape[0]
                    K = self.config.model.quantizer.num_codebooks
                    V = self.config.model.head.num_embeddings

                    with torch.no_grad():
                        self.model.vision_model.eval()
                        vit_feature = self.model.get_vit_feature(x_gen)
                        z_q, code = self.quantizer.get_zq_indices(vit_feature) # z_q: (B, L, 256), code: (B, L, K)
                        abs_code = self.quantizer.to_abs_code(code) # (B, LxK)
                        self.accelerator.print(abs_code.shape)
                        self.accelerator.print(abs_code[0, 0:32])
                    exit(0)


def main(args):
    from omegaconf import OmegaConf
    config = OmegaConf.load(args.config)
    trainer = MyTrainer(config)
    trainer.train()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)