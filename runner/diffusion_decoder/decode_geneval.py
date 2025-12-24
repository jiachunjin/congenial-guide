import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch

@torch.inference_mode()
def decode_code(args):
    from omegaconf import OmegaConf
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from PIL import Image
    from runner.diffusion_decoder.reconstruct import sample_sd3_5
    from runner.diffusion_decoder.sd_decoder import load_mmdit_half_trainable
    from runner.mcq_gen.dev_ar_head import load_quantizer

    # basic configs
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    config = OmegaConf.load(os.path.join(args.exp_dir, "config.yaml"))
    ckpt_path = os.path.join(args.exp_dir, f"model-sd_decoder-{args.step}")
    # load non-trainable models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config.model.sd3_5_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(config.model.sd3_5_path, subfolder="vae")
    vae.requires_grad_(False)
    vae = vae.to(device, dtype).eval()

    quantizer = load_quantizer(config.model.quantizer)
    quantizer.requires_grad_(False)
    quantizer = quantizer.to(device, dtype).eval()

    mmdit = load_mmdit_half_trainable(config.model)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    mmdit.load_state_dict(ckpt, strict=True)
    mmdit = mmdit.to(device, dtype).eval()

    geneval_path = "asset/geneval"
    for sub_dir in os.listdir(geneval_path):
        sub_dir_path = os.path.join(geneval_path, sub_dir)
        code_path = os.path.join(sub_dir_path, "code", "code.pt")
        sample_path = os.path.join(sub_dir_path, "samples")
        code = torch.load(code_path, map_location="cpu").to(device)
        z_q, _ = quantizer.indices_to_feature(code)

        samples = sample_sd3_5(
            transformer         = mmdit,
            vae                 = vae,
            noise_scheduler     = noise_scheduler,
            device              = device,
            dtype               = dtype,
            context             = z_q,
            batch_size          = z_q.shape[0],
            height              = 448,
            width               = 448,
            num_inference_steps = 25,
            guidance_scale      = 1.0,
            seed                = 42
        )
        print(f"{sub_dir} is done")

        os.makedirs(sample_path, exist_ok=True)
        sample_count = 0
        for sample in samples:
            # 将 tensor [C, H, W] 转换为 PIL Image
            sample_pil = Image.fromarray((sample.permute(1, 2, 0).cpu().float().numpy() * 255).astype("uint8"))
            sample_pil.save(os.path.join(sample_path, f"{sample_count:05}.png"))
            sample_count += 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="/inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/experiment/sd_decoder/1206_sd_decoder")
    parser.add_argument("--step", type=int, default=80000)

    args = parser.parse_args()
    decode_code(args)