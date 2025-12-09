import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch

@torch.inference_mode()
def decode_code(args):
    from omegaconf import OmegaConf
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from PIL import Image
    from runner.diffusion_decoder.reconstruct import sample_sd3_5
    from model.internvl.modeling_internvl_chat import InternVLChatModel
    from runner.diffusion_decoder.sd_decoder import load_mmdit_half_trainable
    from runner.mcq_gen.dev_ar_head import load_quantizer
    from runner.diffusion_decoder.sd_decoder import pixel_shuffle
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

    code = torch.load(args.code_path)
    z_q, _ = quantizer.indices_to_feature(code)
    print(z_q.shape)

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
    print(samples.shape)

    import torchvision.utils as vutils
    os.makedirs("asset/code_decode", exist_ok=True)
    
    save_name = args.code_path.split("/")[-1]
    code_decode_path = f"asset/code_decode/{save_name}.png"
    vutils.save_image(samples, code_decode_path, nrow=4, normalize=False)
    print(f"Code decode saved to {code_decode_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--code_path", type=str, required=True)
    args = parser.parse_args()
    decode_code(args)