import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
from util.trainer import Trainer

from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3

def get_sigmas(timesteps, noise_scheduler, device, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

def pixel_shuffle(x, scale_factor=0.5):
    n, w, h, c = x.size()
    # N, W, H, C --> N, W, H * scale, C // scale
    x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
    # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
    x = x.permute(0, 2, 1, 3).contiguous()
    # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
    x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                int(c / (scale_factor * scale_factor)))

    x = x.permute(0, 2, 1, 3).contiguous()

    return x

def load_mmdit_half_trainable(config):
    from safetensors.torch import load_file

    from model.sd_35.mmditx import MMDiTX

    device = torch.device("cpu")
    dtype = torch.bfloat16

    patch_size = 2
    depth = 24
    pos_embed_max_size = 384
    num_patches = 147456
    adm_in_channels = 2048
    qk_norm = "rms"
    x_block_self_attn_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    context_embedder_config = {
        "target": "torch.nn.Linear",
        "params": {
            "in_features": config.context_dim,
            "out_features": 1536,
        },
    }

    transformer = MMDiTX(
        input_size               = None,
        pos_embed_scaling_factor = None,
        pos_embed_offset         = None,
        pos_embed_max_size       = pos_embed_max_size,
        patch_size               = patch_size,
        in_channels              = 16,
        depth                    = depth,
        num_patches              = num_patches,
        adm_in_channels          = adm_in_channels,
        context_embedder_config  = context_embedder_config,
        qk_norm                  = qk_norm,
        x_block_self_attn_layers = x_block_self_attn_layers,
        device                   = device,
        dtype                    = dtype,
        verbose                  = False,
    )

    if config.load_pretrained:
        ckpt = load_file(os.path.join(config.sd3_5_path, "sd3.5_medium.safetensors"))
        new_ckpt = {}
        prefix = "model.diffusion_model."
        for k, v in ckpt.items():
            if k.startswith(prefix):
                new_key = k[len(prefix):]
                new_ckpt[new_key] = v
        del new_ckpt["context_embedder.weight"]
        m, u = transformer.load_state_dict(new_ckpt, strict=False)
        print(f"missing keys: {m}")
        print(f"unexpected keys: {u}")

    # define trainable parameters
    transformer.requires_grad_(False)
    transformer.context_embedder.requires_grad_(True)
    num_para = sum(p.numel() for p in transformer.context_embedder.parameters())
    print("context_embedder parameters: ", num_para / 1e6)

    # name contains "context_block" is trainable
    for name, param in transformer.named_parameters():
        if "context_block" in name:
            param.requires_grad_(True)

    num_para = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print("total parameters: ", num_para / 1e6)
    

    return transformer

class MyTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

    def _load_models(self):
        from transformers import AutoTokenizer
        from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
        from model.internvl.modeling_internvl_chat import InternVLChatModel
        from runner.mcq_gen.dev_ar_head import load_quantizer

        clip = InternVLChatModel.from_pretrained(self.config.model.internvl_path).vision_model
        clip.requires_grad_(False)
        quantizer = load_quantizer(self.config.model.quantizer)
        vae = AutoencoderKL.from_pretrained(self.config.model.sd3_5_path, subfolder="vae")
        vae.requires_grad_(False)
        mmdit = load_mmdit_half_trainable(self.config.model)
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.config.model.sd3_5_path, subfolder="scheduler")

        if self.config.train.resume_path is not None:
            raise NotImplementedError("Resume training is not supported for this script")

        self.vae = vae.to(self.device, self.dtype).eval()
        self.clip = clip.to(self.device, self.dtype).eval()
        self.quantizer = quantizer.to(self.device, self.dtype).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.internvl_path, trust_remote_code=True, use_fast=False)
        self.noise_scheduler = noise_scheduler
        self.model = mmdit

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
                    pixel_values_clip = (pixel_values - imagenet_mean) / imagenet_std
                    pixel_values_vae = pixel_values * 2 - 1
                    with torch.no_grad():
                        vit_embeds = self.clip(
                            pixel_values         = pixel_values_clip,
                            output_hidden_states = False,
                            return_dict          = True).last_hidden_state[:, 1:, :]

                        h = w = int(vit_embeds.shape[1] ** 0.5)
                        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
                        vit_embeds = pixel_shuffle(vit_embeds, scale_factor=0.5)
                        x_clip = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
                        x_clip_q = self.quantizer.get_z_q(x_clip)
                        x_vae = self.vae.encode(pixel_values_vae).latent_dist.sample()

                    print(x_clip.shape, x_clip_q.shape, x_vae.shape)

                    model_input = (x_vae - self.vae.config.shift_factor) * self.vae.config.scaling_factor
                    noise = torch.randn_like(model_input)

                    u = compute_density_for_timestep_sampling(
                        weighting_scheme = "logit_normal",
                        batch_size       = model_input.shape[0],
                        logit_mean       = 0.0,
                        logit_std        = 1.0,
                        mode_scale       = 1.29,
                    )
                    indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
                    timesteps = self.noise_scheduler.timesteps[indices].to(device=model_input.device)
                    sigmas = get_sigmas(timesteps, self.noise_scheduler, self.device, n_dim=model_input.ndim, dtype=model_input.dtype)
                    noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                    model_pred = self.model(
                        x           = noisy_model_input,
                        t           = timesteps,
                        context     = x_clip_q,
                        y           = None,
                    )

                    model_pred = model_pred * (-sigmas) + noisy_model_input
                    weighting = compute_loss_weighting_for_sd3(weighting_scheme="logit_normal", sigmas=sigmas)
                    target = model_input

                    loss = torch.mean((weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), 1).mean()

                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.params_to_learn, 1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        self.global_step += 1
                        self.progress_bar.update(1)
                        logs = dict(
                            loss_fm = self.accelerator.gather(loss.detach()).mean().item(),
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