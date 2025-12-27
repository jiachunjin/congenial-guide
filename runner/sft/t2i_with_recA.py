import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
from einops import rearrange
from util.trainer import Trainer
from runner.mcq_gen.dev_ar_head import load_quantizer
from runner.mixture_modality.moe import modify_internvl_to_mixture


class MyTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

    def _load_models(self):
        from transformers import AutoTokenizer
        from model.internvl.modeling_internvl_chat import InternVLChatModel

        quantizer = load_quantizer(self.config.model.quantizer)
        internvl = InternVLChatModel.from_pretrained(self.config.model.internvl_path)
        internvl = modify_internvl_to_mixture(internvl, self.config.model)

        if self.config.train.resume_path is not None:
            ckpt_path = os.path.join(self.config.train.resume_path, "pytorch_model/mp_rank_00_model_states.pt")
            ckpt = torch.load(ckpt_path, map_location="cpu")["module"]
            m, u = internvl.load_state_dict(ckpt, strict=False)
            print(f"unused keys: {u}")

        tokenizer = AutoTokenizer.from_pretrained(self.config.model.internvl_path, trust_remote_code=True, use_fast=False)

        self.model = internvl
        self.quantizer = quantizer.to(self.device, self.dtype).eval()
        self.tokenizer = tokenizer

    def _load_dataloader(self):
        from util.dataloader import get_jackyhate_dataloader
        self.dataloader = get_jackyhate_dataloader(self.config.data, self.tokenizer)

    def train(self):
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(self.model, self.optimizer, self.scheduler)

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        imagenet_mean = torch.tensor(IMAGENET_MEAN, device=self.accelerator.device, dtype=self.dtype).view(1, 3, 1, 1)
        imagenet_std = torch.tensor(IMAGENET_STD, device=self.accelerator.device, dtype=self.dtype).view(1, 3, 1, 1)
        training_done = False

        IMG_START_TOKEN = "<img>"
        IMG_END_TOKEN = "</img>"
        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        prompt = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * 256 + IMG_END_TOKEN + "\n" + "Describe this image in detail." + IMG_START_TOKEN

        tokenizer_output = self.tokenizer(
            [prompt],
            return_tensors = "pt",
            padding        = "max_length",
            padding_side   = "left",
            truncation     = True,
            max_length     = self.config.data.max_seq_length - self.config.data.num_img_token,
        )
        rec_input_ids = torch.LongTensor(tokenizer_output["input_ids"]).to(self.device)
        rec_attention_mask = torch.LongTensor(tokenizer_output["attention_mask"]).to(self.device)

        while not training_done:
            for batch in self.dataloader:
                with self.accelerator.accumulate(self.model):
                    self.model.train()
                    self.model.vision_model.eval()  # 确保 vision_model 保持 eval

                    pixel_values = batch["pixel_values"].to(self.device, self.dtype)
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)

                    x_gen = (pixel_values - imagenet_mean) / imagenet_std

                    print(rec_input_ids.shape, input_ids.shape)
                    print(rec_attention_mask.shape, attention_mask.shape)
                    exit(0)
                    break

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