import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
from accelerate import Accelerator

@torch.inference_mode()
def decode_code(args):
    from omegaconf import OmegaConf
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from PIL import Image
    from runner.diffusion_decoder.reconstruct import sample_sd3_5
    from runner.diffusion_decoder.sd_decoder import load_mmdit_half_trainable
    from runner.mcq_gen.dev_ar_head import load_quantizer

    # basic configs
    accelerator = Accelerator()
    device = accelerator.device
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

    geneval_path = args.geneval_path
    sub_dirs = sorted(os.listdir(geneval_path))
    
    # 将任务分配到不同的GPU
    num_processes = accelerator.num_processes
    process_index = accelerator.process_index
    
    for index, sub_dir in enumerate(sub_dirs):
        # 只处理分配给当前进程的任务
        if index % num_processes != process_index:
            continue
            
        sub_dir_path = os.path.join(geneval_path, sub_dir)
        code_path = os.path.join(sub_dir_path, "code", "code.pt")
        sample_path = os.path.join(sub_dir_path, "samples")
        # if sample_path is not empty, skip
        if os.path.exists(sample_path) and len(os.listdir(sample_path)) > 0:
            continue
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
        accelerator.print(f"Process {process_index}: {sub_dir} ({index: >3}/{len(sub_dirs)}) is done")

        os.makedirs(sample_path, exist_ok=True)
        sample_count = 0
        for sample in samples:
            # 将 tensor [C, H, W] 转换为 PIL Image
            sample_pil = Image.fromarray((sample.permute(1, 2, 0).cpu().float().numpy() * 255).astype("uint8"))
            sample_pil.save(os.path.join(sample_path, f"{sample_count:05}.png"))
            sample_count += 1

    # 等待所有进程完成
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="/inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/experiment/sd_decoder/1206_sd_decoder")
    parser.add_argument("--step", type=int, default=80000)
    parser.add_argument("--geneval_path", type=str, required=True)

    args = parser.parse_args()
    decode_code(args)