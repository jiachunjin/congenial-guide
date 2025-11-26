import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from einops import rearrange
from util.trainer import Trainer

class MyTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
    
    def _load_models(self):
        from model.janus.models import MultiModalityCausalLM, VLChatProcessor

        janus = MultiModalityCausalLM.from_pretrained(self.config.model.pretrained_path)
        vl_chat_processor = VLChatProcessor.from_pretrained(self.config.model.pretrained_path)

        self.model = janus
        self.preprocessor = vl_chat_processor
    
    def _load_dataloader(self):
        from util.dataloader import get_blip3o_dataloader_janus
        self.dataloader = get_blip3o_dataloader_janus(self.config.data, self.preprocessor, self.accelerator)
    
    def train(self):
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        training_done = False
        while not training_done:
            for batch in self.dataloader:
                with self.accelerator.accumulate(self.model):
                    self.model.train()

                    pixel_values = batch["pixel_values"].to(self.device, self.dtype)
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)

                    x_gen = pixel_values * 2 - 1
                    B = x_gen.shape[0]

                    with torch.no_grad():
                        _, _, info = self.model.gen_vision_model.encode(x_gen)
                        code = rearrange(info[2], "(B L) -> B L", B=B)
                    
                    text_embedding = self.model.language_model.get_input_embeddings()(input_ids)
                    visual_embedding = self.model.prepare_gen_img_embeds(code)
                    # print(text_embedding.shape, visual_embedding.shape)
                    # exit(0)
                    joint_embedding = torch.cat([text_embedding, visual_embedding], dim=1)
                    attention_mask = torch.cat([attention_mask, torch.ones((B, self.config.data.num_img_token), dtype=torch.bool, device=self.device)], dim=1)

                    visual_hidden_states = self.model.language_model(
                        inputs_embeds        = joint_embedding,
                        attention_mask       = attention_mask,
                        output_hidden_states = True,
                    ).hidden_states[-1][:, -self.config.data.num_img_token-1:-1, :]

                    visual_token_logits = self.model.gen_head(visual_hidden_states)

                    loss = torch.nn.functional.cross_entropy(visual_token_logits.contiguous().view(-1, visual_token_logits.size(-1)), code.contiguous().view(-1))
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
                            print(f"model saved to {save_path}")

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