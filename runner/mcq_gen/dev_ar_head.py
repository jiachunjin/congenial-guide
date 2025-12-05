import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
from util.trainer import Trainer

def modify_internvl(internvl, config):
    """
    add ar_head and visual embeddings to internvl
    """
    from model.mcq_gen.ar_head import ARHead
    ar_head = ARHead(config.ar_head)
    ar_head.requires_grad_(True)

    internvl.ar_head = ar_head

    internvl.visual_projector = nn.Sequential(
        nn.LayerNorm(config.embedding_dim, eps=1e-6),
        nn.Linear(config.embedding_dim, config.llm_hidden_size),
        nn.GELU(),
        nn.Linear(config.llm_hidden_size, config.llm_hidden_size),
    )

    num_params = sum(p.numel() for p in internvl.ar_head.parameters() if p.requires_grad)
    print(f"Trainable parameters in ar_head: {num_params / 1e6:.2f}M")

    num_params = sum(p.numel() for p in internvl.visual_projector.parameters() if p.requires_grad)
    print(f"Trainable parameters in visual_projector: {num_params / 1e6:.2f}M")

    return internvl

def load_quantizer(config):
    from model.quantizer.lfq import get_lfq_quantizer
    from model.quantizer.vq import get_vq_quantizer
    from model.quantizer.multi_vq import get_multi_vq_quantizer

    vq_type = getattr(config, "vq_type", "lfq")
    if vq_type == "lfq":
        clip_quantizer = get_lfq_quantizer(config)
    elif vq_type == "vq":
        clip_quantizer = get_vq_quantizer(config)
    elif vq_type == "multi_vq":
        clip_quantizer = get_multi_vq_quantizer(config)
    else:
        raise ValueError(f"Invalid VQ type: {vq_type}")

    if config.ckpt_path is not None:
        ckpt = torch.load(config.ckpt_path, map_location="cpu", weights_only=True)
        clip_quantizer.load_state_dict(ckpt, strict=True)
        print(f"clip_quantizer loaded from {config.ckpt_path}")
    else:
        print("Attention !!! For development only")

    clip_quantizer.requires_grad_(False)

    return clip_quantizer

class MyTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

    def _load_models(self):
        """
        这个script的目的主要是为了测试ar_head以及multi-code token input, 所以ar backbone只用最naive的InternVL
        """
        from transformers import AutoTokenizer
        from model.internvl.modeling_internvl_chat import InternVLChatModel

        quantizer = load_quantizer(self.config.model.quantizer)

        internvl = InternVLChatModel.from_pretrained(self.config.model.internvl_path)
        internvl = modify_internvl(internvl, self.config.model)

        if self.config.train.resume_path is not None:
            raise NotImplementedError("Resume training is not supported for this script")

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

                    with torch.no_grad():
                        vit_feature = self.model.get_vit_feature(x_gen)
                        _, code, _ = self.quantizer(vit_feature) # code: (B, L, K)
                        labels = code.permute(0, 2, 1).contiguous().view(-1).long()
                        visual_features, _ = self.quantizer.indices_to_feature(code)
                        text_embedding_t2i = self.model.language_model.get_input_embeddings()(input_ids)

                    visual_embedding_t2i = self.model.visual_projector(visual_features)
                    joint_embedding = torch.cat([text_embedding_t2i, visual_embedding_t2i], dim=1)
                    attention_mask = torch.cat([attention_mask, torch.ones((B, self.config.data.num_img_token), dtype=torch.bool, device=self.device)], dim=1)

                    base_tokens = self.model.language_model(
                        inputs_embeds        = joint_embedding,
                        attention_mask       = attention_mask,
                        output_hidden_states = True,
                    ).hidden_states[-1][:, -self.config.data.num_img_token-1:-1, :]

                    B, L, D = base_tokens.shape
                    base_tokens = base_tokens.reshape(B * L, 1, D)
                    targets = code.reshape(B * L, K)[:, :-1] # (BxL, K-1)
                    index_embeddings = []
                    for i in range(K - 1):
                        index_embed = self.model.ar_head.codebooks[i](targets[:, i])
                        index_embeddings.append(index_embed)
                    index_embeddings = torch.stack(index_embeddings, dim=1)
                    self.accelerator.print(f"index_embeddings.shape: {index_embeddings.shape}")
                    h = torch.cat((base_tokens, index_embeddings), dim=1)  # [B*L, K, C]
                    self.accelerator.print(f"h.shape: {h.shape}")
                    logits = self.model.ar_head(h)
                    logits = logits.reshape(B, L, K, -1).permute(0, 2, 1, 3)  # [B, K, L, sub_vocab_size]
                    logits = logits.reshape(-1, self.config.model.ar_head.num_embeddings)
                    self.accelerator.print(f"logits.shape: {logits.shape}")
                    loss_fct = torch.nn.CrossEntropyLoss()
                    loss = loss_fct(logits, labels)
                    self.accelerator.print(f"loss: {loss.item()}")
                    
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
                            state_dict = self.accelerator.unwrap_model(self.model.clip_quantizer).state_dict()
                            save_path = os.path.join(self.output_dir, f"quantizer-{self.config.train.exp_name}-{self.global_step}")
                            torch.save(state_dict, save_path)
                            print(f"Quantizer saved to {save_path}")

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