import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import copy
from util.trainer import Trainer


def add_quantizer(internvl, config):
    from model.quantizer.lfq import get_lfq_quantizer

    clip_quantizer = get_lfq_quantizer(config)
    clip_quantizer.requires_grad_(True)
    internvl.clip_quantizer = clip_quantizer
    
    num_params = sum(p.numel() for p in internvl.clip_quantizer.parameters() if p.requires_grad)
    print(f"Trainable parameters in clip_quantizer: {num_params}")

    return internvl

class MyTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

    def _load_models(self):
        from transformers import AutoTokenizer
        from model.internvl.modeling_internvl_chat import InternVLChatModel

        internvl = InternVLChatModel.from_pretrained(self.config.model.internvl_path)
        internvl.requires_grad_(False)
        teacher = copy.deepcopy(internvl)
        internvl = add_quantizer(internvl, self.config.model.quantizer)

        if self.config.train.resume_path is not None:
            ckpt = torch.load(self.config.train.resume_path, map_location="cpu", weights_only=True)
            m, u = internvl.clip_quantizer.load_state_dict(ckpt, strict=False)
            print(f"missing keys: {m}, unmatched keys: {u}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.internvl_path, trust_remote_code=True, use_fast=False)
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.model = internvl
        self.teacher = teacher.to(self.device, self.dtype).eval()

    def _load_dataloader(self):
        from util.dataloader_llava import get_llava_mix665k_dataloader

        self.dataloader = get_llava_mix665k_dataloader(self.config.data, self.tokenizer)

    def train(self):
        self.model, self.optimizer, self.dataloader = self.accelerator.prepare(self.model, self.optimizer, self.dataloader)

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
                    loss = kl_div

                    # clip_mse = torch.nn.functional.mse_loss(x_vq, vit_embeds_teacher)
                    # clip_cosine = 1 - torch.nn.functional.cosine_similarity(x_vq, vit_embeds_teacher, dim=-1).mean()

                    # loss = kl_div + self.config.train.hp_mse * clip_mse + self.config.train.hp_cosine * clip_cosine

                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.params_to_learn, 1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        self.global_step += 1
                        self.progress_bar.update(1)
                        logs = dict(
                            loss_und = self.accelerator.gather(kl_div.detach()).mean().item(),
                        )
                        # if self.config.train.hp_mse != 0:
                        #     logs["clip_mse"] = self.accelerator.gather(clip_mse.detach()).mean().item(),
                        # if self.config.train.hp_cosine != 0:
                        #     logs["clip_cosine"] = self.accelerator.gather(clip_cosine.detach()).mean().item(),

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