import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
from einops import rearrange
from util.trainer import Trainer
from runner.mcq_gen.dev_ar_head import load_quantizer

def modify_internvl_my_ar_head(internvl, config):
    """
    add my ar head to internvl
    """
    from model.mcq_gen.my_ar_head import MyARHead
    
    visual_projector = nn.Sequential(
        nn.Linear(config.embedding_dim, config.llm_hidden_size),
        nn.GELU(),
        nn.Linear(config.llm_hidden_size, config.llm_hidden_size),
    )
    visual_projector.requires_grad_(True)
    num_params = sum(p.numel() for p in visual_projector.parameters() if p.requires_grad)
    print(f"Trainable parameters in visual_projector: {num_params / 1e6:.2f}M")
    
    ar_head = MyARHead(config.head)
    ar_head.requires_grad_(True)
    num_params = sum(p.numel() for p in ar_head.parameters() if p.requires_grad)
    print(f"Trainable parameters in ar_head: {num_params / 1e6:.2f}M")

    internvl.visual_projector = visual_projector
    internvl.ar_head = ar_head

    return internvl


class MyTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

    def _load_models(self):
        from transformers import AutoTokenizer
        from model.internvl.modeling_internvl_chat import InternVLChatModel

        quantizer = load_quantizer(self.config.model.quantizer)
        internvl = InternVLChatModel.from_pretrained(self.config.model.internvl_path)
        internvl = modify_internvl_my_ar_head(internvl, self.config.model)

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

                    with torch.no_grad():
                        self.model.vision_model.eval()
                        vit_feature = self.model.get_vit_feature(x_gen)
                        z_q, code = self.quantizer.get_zq_indices(vit_feature) # z_q: (B, L, 256), code: (B, L, K)

                    B, L, K = code.shape
                    V = self.config.model.head.num_embeddings

                    text_embedding_t2i = self.model.language_model.get_input_embeddings()(input_ids)
                    visual_embedding_t2i = self.model.visual_projector(z_q)
                    joint_embedding = torch.cat([text_embedding_t2i, visual_embedding_t2i], dim=1)
                    attention_mask = torch.cat([attention_mask, torch.ones((B, self.config.data.num_img_token), dtype=torch.bool, device=self.device)], dim=1)

                    visual_hidden_states = self.model.language_model(
                        inputs_embeds        = joint_embedding,
                        attention_mask       = attention_mask,
                        output_hidden_states = True,
                    ).hidden_states[-1][:, -self.config.data.num_img_token-1:-1, :] # (B, L, D)

                    prefix = rearrange(visual_hidden_states, "B L D -> (B L) 1 D")
                    head_visual_embeddings = self.model.ar_head._code_to_embeddings(code) # (BxL, K, D)
                    h = torch.cat((prefix, head_visual_embeddings), dim=1) # (BxL, K+1, D)

                    logits = self.model.ar_head(h[:, :-1, :]) # (BxL, K, V)
                    logits = rearrange(logits, "(B L) K V -> B L K V", B=B, L=L)
                    loss = torch.nn.functional.cross_entropy(logits.view(-1, V), code.view(-1))

                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.params_to_learn, 1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        self.global_step += 1
                        self.progress_bar.update(1)
                        logs = dict(
                            loss_CE = self.accelerator.gather(loss.detach()).mean().item(),
                        )

                        self.accelerator.log(logs, step=self.global_step)
                        self.progress_bar.set_postfix(**logs)

                        if self.global_step > 0 and self.global_step % self.config.train.save_every == 0 and self.accelerator.is_main_process:
                            self.model.eval()
                            state_dict = self.accelerator.unwrap_model(self.model).state_dict()
                            save_path = os.path.join(self.output_dir, f"model-{self.config.train.exp_name}-{self.global_step}")
                            torch.save(state_dict, save_path)
                            print(f"Model saved to {save_path}")

                        self.accelerator.wait_for_everyone()

                        if self.global_step >= self.config.train.num_iter:
                            training_done = True
                            break

            self.epoch += 1
            self.accelerator.print(f"epoch {self.epoch}: finished")
            self.accelerator.log({"epoch": self.epoch}, step=self.global_step)

        self.accelerator.end_training()

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