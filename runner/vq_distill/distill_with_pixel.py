import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import copy
from util.trainer import Trainer
from runner.vq_distill.distill import add_quantizer


class MyTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

    def _load_models(self):
        from transformers import AutoTokenizer
        from model.internvl.modeling_internvl_chat import InternVLChatModel
        from model.pixel_decoder.vit_decoder import ViT_Decoder
        from model.pixel_decoder.rec_loss import RecLoss

        internvl = InternVLChatModel.from_pretrained(self.config.model.internvl_path)
        internvl.requires_grad_(False)
        teacher = copy.deepcopy(internvl)
        internvl = add_quantizer(internvl, self.config.model.quantizer)

        pixel_decoder = ViT_Decoder(self.config.model.pixel_decoder)
        rec_loss = RecLoss(self.config.rec_loss)
        internvl.pixel_decoder = pixel_decoder
        internvl.rec_loss = rec_loss

        if self.config.train.resume_path is not None:
            ckpt = torch.load(self.config.train.resume_path, map_location="cpu", weights_only=True)
            internvl.clip_quantizer.load_state_dict(ckpt["quantizer"], strict=True)
            print(f"clip_quantizer loaded from {self.config.train.resume_path}")

            internvl.rec_loss.load_state_dict(ckpt["rec_loss"], strict=True)
            print(f"rec_loss loaded from {self.config.train.resume_path}")

            internvl.pixel_decoder.load_state_dict(ckpt["pixel_decoder"], strict=True)
            print(f"pixel_decoder loaded from {self.config.train.resume_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.internvl_path, trust_remote_code=True, use_fast=False)
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.model = internvl
        self.teacher = teacher.to(self.device, self.dtype).eval()
    
    def _load_optimizer(self):
        self.params_to_learn = list(p for p in self.model.pixel_decoder.parameters() if p.requires_grad) + list(p for p in self.model.clip_quantizer.parameters() if p.requires_grad)
        self.optimizer = torch.optim.AdamW(
            self.params_to_learn,
            lr           = self.config.train.lr,
            betas        = (0.9, 0.95),
            weight_decay = 5e-2,
            eps          = 1e-8,
        )
        self.disc_params = list(self.model.rec_loss.parameters())
        self.optimizer_disc = torch.optim.AdamW(
            self.disc_params,
            lr           = self.config.train.lr_disc,
            betas        = (0.9, 0.95),
            weight_decay = 5e-2,
            eps          = 1e-8,
        )

    def _load_dataloader(self):
        from util.dataloader_llava import get_llava_mix665k_dataloader

        self.dataloader = get_llava_mix665k_dataloader(self.config.data, self.tokenizer)

    def train(self):
        self.model, self.optimizer, self.optimizer_disc, self.dataloader = self.accelerator.prepare(self.model, self.optimizer, self.optimizer_disc. self.dataloader)

        training_done = False
        while not training_done:
            for batch in self.dataloader:
                with self.accelerator.accumulate(self.model):
                    self.model.train()
                    if batch is None:
                        print("Batch is None, 出问题了!!!")
                        continue

                    pixel_values = batch["pixel_values"].to(self.dtype)
                    input_ids = batch["input_ids"].to(torch.int64)
                    attention_mask = batch["attention_mask"].to(torch.bool)
                    answer_mask = batch["answer_mask"].to(torch.bool)

                    # ---------- get visual feature ----------
                    with torch.no_grad():
                        vit_feature = self.model.get_vit_feature(pixel_values)
                    x_vq, code = self.model.clip_quantizer(vit_feature)
                    vit_embeds_teacher = self.teacher.mlp1(vit_feature)

                    # ---------- build input embeddings for teacher and model ----------
                    input_embeds_teacher = self.teacher.language_model.get_input_embeddings()(input_ids)
                    B, N, C = input_embeds_teacher.shape
                    input_embeds_teacher = input_embeds_teacher.reshape(B * N, C)

                    input_ids = input_ids.reshape(B * N)
                    selected = (input_ids == self.img_context_token_id)
                    assert selected.sum() != 0
                    input_embeds_student = input_embeds_teacher.clone()
                    input_embeds_student[selected] = x_vq.reshape(-1, C).to(input_embeds_student.device)
                    input_embeds_teacher[selected] = vit_embeds_teacher.reshape(-1, C).to(input_embeds_teacher.device)

                    input_embeds_student = input_embeds_student.reshape(B, N, C)
                    input_embeds_teacher = input_embeds_teacher.reshape(B, N, C)

                    # ---------- compute understanding distillation loss ----------
                    student_outputs = self.model.language_model(
                        inputs_embeds        = input_embeds_student,
                        attention_mask       = attention_mask,
                        output_hidden_states = True,
                    )
                    answer_logits_student = student_outputs.logits[answer_mask]
                    hidden_states_student = student_outputs.hidden_states[-1]

                    B, N, C = hidden_states_student.shape
                    hidden_states_student = hidden_states_student.reshape(B * N, C)
                    hidden_states_student = hidden_states_student[selected]
                    hidden_states_student = hidden_states_student.reshape(B, 256, C)

                    answer_logits_teacher = self.teacher.language_model(
                        inputs_embeds        = input_embeds_teacher,
                        attention_mask       = attention_mask,
                        output_hidden_states = False,
                    ).logits[answer_mask]

                    answer_logits_student_log_softmax = torch.nn.functional.log_softmax(answer_logits_student, dim=-1)
                    answer_logits_teacher_log_softmax = torch.nn.functional.log_softmax(answer_logits_teacher, dim=-1)
                    kl_div = torch.nn.functional.kl_div(answer_logits_student_log_softmax, answer_logits_teacher_log_softmax, log_target=True, reduction="batchmean")

                    # ---------- compute pixel reconstruction loss ----------
                    rec = self.model.pixel_decoder(x_vq)
                    loss_rec, loss_rec_dict = self.model.rec_loss(pixel_values * 2 - 1, rec, self.global_step, "generator")

                    loss = kl_div + self.config.train.hp_rec * loss_rec

                    # ---------- train discriminator ----------
                    loss_disc, loss_disc_dict = self.model.rec_loss(pixel_values * 2 - 1, rec, self.global_step, "discriminator")

                    self.accelerator.backward(loss_disc)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.params_to_learn, 1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        self.accelerator.clip_grad_norm_(self.disc_params, 1.0)
                        self.optimizer_disc.step()
                        self.optimizer_disc.zero_grad()

                        self.global_step += 1
                        self.progress_bar.update(1)
                        logs = dict(
                            loss_und = self.accelerator.gather(kl_div.detach()).mean().item(),
                            **loss_rec_dict,
                            **loss_disc_dict,
                        )
                        self.accelerator.log(logs, step=self.global_step)
                        self.progress_bar.set_postfix(**logs)

                        if self.global_step > 0 and self.global_step % self.config.train.save_every == 0 and self.accelerator.is_main_process:
                            self.model.eval()
                            quantizer_state_dict = self.accelerator.unwrap_model(self.model.clip_quantizer).state_dict()
                            rec_loss_state_dict = self.accelerator.unwrap_model(self.model.rec_loss).state_dict()
                            pixel_decoder_state_dict = self.accelerator.unwrap_model(self.model.pixel_decoder).state_dict()
                            save_path = os.path.join(self.output_dir, f"ckpt-{self.config.train.exp_name}-{self.global_step}")
                            state_dict = {
                                "quantizer": quantizer_state_dict,
                                "rec_loss": rec_loss_state_dict,
                                "pixel_decoder": pixel_decoder_state_dict,
                            }
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