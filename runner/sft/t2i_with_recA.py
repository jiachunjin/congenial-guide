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

                    with torch.no_grad():
                        vit_feature = self.model.get_vit_feature(x_gen)
                        z_q, code = self.quantizer.get_zq_indices(vit_feature)
                        original_visual_embedding = self.model.mlp1(vit_feature)

                    B, L, _ = code.shape
                    V = self.config.model.head.num_embeddings
                    # ----- prepare RecA textual embeddings -----
                    text_embedding_rec = self.model.language_model.get_input_embeddings()(rec_input_ids).repeat(B, 1, 1)
                    B, N, C = text_embedding_rec.shape
                    text_embedding_rec = text_embedding_rec.reshape(B * N, C)
                    img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
                    rec_input_ids_flatten = rec_input_ids.repeat(B, 1).reshape(B * N)
                    selected = (rec_input_ids_flatten == img_context_token_id)
                    assert selected.sum() != 0
                    text_embedding_rec[selected] = original_visual_embedding.reshape(-1, C).to(self.device)
                    text_embedding_rec = text_embedding_rec.reshape(B, N, C)
                    visual_embedding_t2i = self.model.visual_projector(z_q)

                    joint_embedding_rec = torch.cat([text_embedding_rec, visual_embedding_t2i], dim=1)
                    attention_mask_rec = torch.cat([rec_attention_mask.repeat(B, 1), torch.ones((B, self.config.data.num_img_token), dtype=torch.bool, device=self.device)], dim=1)

                    text_embedding_t2i = self.model.language_model.get_input_embeddings()(input_ids)
                    visual_embedding_t2i = self.model.visual_projector(z_q)
                    joint_embedding_t2i = torch.cat([text_embedding_t2i, visual_embedding_t2i], dim=1)
                    attention_mask_t2i = torch.cat([attention_mask, torch.ones((B, self.config.data.num_img_token), dtype=torch.bool, device=self.device)], dim=1)

                    L_txt = text_embedding_t2i.shape[1]
                    L_visual = visual_embedding_t2i.shape[1]
                    B_orig = B
                    B = B * 2
                    code = code.repeat(2, 1, 1)
                    vision_token_mask = torch.cat([torch.zeros(B, L_txt-1), torch.ones(B, L_visual+1)], dim=1).to(self.device, dtype=self.dtype)
                    visual_hidden_states = self.model.language_model(
                        inputs_embeds        = torch.cat([joint_embedding_rec, joint_embedding_t2i], dim=0),
                        attention_mask       = torch.cat([attention_mask_rec, attention_mask_t2i], dim=0),
                        vision_token_mask    = vision_token_mask,
                        output_hidden_states = True,
                    ).hidden_states[-1][:, -self.config.data.num_img_token-1:-1, :] # (B, L, D)

                    prefix = rearrange(visual_hidden_states, "B L D -> (B L) 1 D")
                    head_visual_embeddings = self.model.ar_head._code_to_embeddings(code) # (BxL, K, D)
                    h = torch.cat((prefix, head_visual_embeddings), dim=1) # (BxL, K+1, D)

                    logits = self.model.ar_head(h[:, :-1, :]) # (BxL, K, V)
                    logits = rearrange(logits, "(B L) K V -> B L K V", B=B, L=L)
                    
                    # 切分为 REC 和 T2I 两部分
                    logits_rec, logits_t2i = logits[:B_orig], logits[B_orig:]
                    code_rec, code_t2i = code[:B_orig], code[B_orig:]
                    
                    loss_rec = torch.nn.functional.cross_entropy(logits_rec.reshape(-1, V), code_rec.reshape(-1))
                    loss_t2i = torch.nn.functional.cross_entropy(logits_t2i.reshape(-1, V), code_t2i.reshape(-1))
                    loss = (loss_rec + loss_t2i) / 2

                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(self.params_to_learn, 1.0)
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()

                        self.global_step += 1
                        self.progress_bar.update(1)

                        logs = dict(
                            loss_CE = self.accelerator.gather(loss.detach()).mean().item(),
                            loss_rec = self.accelerator.gather(loss_rec.detach()).mean().item(),
                            loss_t2i = self.accelerator.gather(loss_t2i.detach()).mean().item(),
                            lr = self.scheduler.get_last_lr()[0],
                            grad_norm = grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm,
                        )
                        self.accelerator.log(logs, step=self.global_step)
                        self.progress_bar.set_postfix(**logs)

                        if self.global_step > 0 and self.global_step % self.config.train.save_every == 0:
                            self.accelerator.wait_for_everyone()
                            self.accelerator.save_state(os.path.join(self.output_dir, f"checkpoint-{self.global_step}"))

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