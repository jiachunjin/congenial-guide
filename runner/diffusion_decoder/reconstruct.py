import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
@torch.no_grad()
def sample_sd3_5(
    transformer,
    vae,
    noise_scheduler,
    device,
    dtype, 
    context,
    batch_size          = 1,
    height              = 192,
    width               = 192,
    num_inference_steps = 20,
    guidance_scale      = 1.0,
    seed                = None,
    multi_modal_context = False,
):
    from tqdm import tqdm
    if seed is not None:
        torch.manual_seed(seed)
    
    transformer.eval()
    
    latent_height = height // 8
    latent_width = width // 8
    
    latents = torch.randn(
        (batch_size, 16, latent_height, latent_width),
        device = device,
        dtype  = dtype
    )

    noise_scheduler.set_timesteps(num_inference_steps)
    timesteps = noise_scheduler.timesteps.to(device=device, dtype=dtype)
    
    for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
        if t.ndim == 0:
            t = t.unsqueeze(0)
        t = t.repeat(batch_size)

        latent_model_input = latents

        if guidance_scale > 1.0:
            latent_model_input = torch.cat([latent_model_input, latent_model_input], dim=0)
            t = torch.cat([t, t], dim=0)
            context_ = torch.cat([context, torch.zeros_like(context, device=context.device, dtype=context.dtype)], dim=0)
        else:
            context_ = context

        noise_pred = transformer(
            x           = latent_model_input,
            t           = t,
            context     = context_,
            y           = None,
            multi_modal_context = multi_modal_context,
        )

        if guidance_scale > 1.0:
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        step_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=t[0] if t.ndim > 0 else t,
            sample=latents,
            return_dict=False,
        )
        latents = step_output[0]
    
    latents = 1 / vae.config.scaling_factor * latents + vae.config.shift_factor
    image = vae.decode(latents).sample
    
    image = (image / 2 + 0.5).clamp(0, 1)
    
    return image

@torch.inference_mode()
def reconstruct(args):
    from omegaconf import OmegaConf
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
    import torchvision.transforms as pth_transforms
    from PIL import Image

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
    clip = InternVLChatModel.from_pretrained(config.model.internvl_path).vision_model
    clip.requires_grad_(False)
    quantizer = load_quantizer(config.model.quantizer)
    quantizer.requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(config.model.sd3_5_path, subfolder="vae")
    vae.requires_grad_(False)
    clip = clip.to(device, dtype).eval()
    quantizer = quantizer.to(device, dtype).eval()
    vae = vae.to(device, dtype).eval()
    # load trained SD decoder
    mmdit = load_mmdit_half_trainable(config.model)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    mmdit.load_state_dict(ckpt, strict=True)
    mmdit = mmdit.to(device, dtype).eval()

    vae_transform = pth_transforms.Compose([
        pth_transforms.Resize(448, max_size=None),
        pth_transforms.CenterCrop(448),
        pth_transforms.ToTensor(),
    ])

    images = [
        Image.open("/inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/codebase/congenial-guide/asset/img/messi.webp").convert("RGB"),
        Image.open("/inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/codebase/congenial-guide/asset/img/非常厉害.png").convert("RGB"),
        Image.open("/inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/codebase/congenial-guide/asset/img/shapes.png").convert("RGB"),
        Image.open("/inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/codebase/congenial-guide/asset/img/english.png").convert("RGB"),
        Image.open("/inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/codebase/congenial-guide/asset/img/ad.jpeg").convert("RGB"),
        Image.open("/inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/codebase/congenial-guide/asset/img/ad_eng.webp").convert("RGB"),
        Image.open("/inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/codebase/congenial-guide/asset/img/jay.jpg").convert("RGB"),
        Image.open("/inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/codebase/congenial-guide/asset/img/jay2.jpeg").convert("RGB"),
        Image.open("/inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/codebase/congenial-guide/asset/img/book.png").convert("RGB"),
        Image.open("/inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/codebase/congenial-guide/asset/img/norris.png").convert("RGB"),
        Image.open("/inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/codebase/congenial-guide/asset/img/taylor.jpg").convert("RGB"),
        Image.open("/inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/codebase/congenial-guide/asset/img/usa.jpg").convert("RGB"),
    ]
    x_list = []
    for img in images:
        x_list.append(vae_transform(img).unsqueeze(0).to(device, dtype))
    x = torch.cat(x_list, dim=0)

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    imagenet_mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    imagenet_std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    x = (x - imagenet_mean) / imagenet_std

    vit_embeds = clip(
        pixel_values         = x,
        output_hidden_states = False,
        return_dict          = True).last_hidden_state[:, 1:, :]

    h = w = int(vit_embeds.shape[1] ** 0.5)
    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
    vit_embeds = pixel_shuffle(vit_embeds, scale_factor=0.5)
    x_clip = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
    z_q, _ = quantizer.get_zq_indices(x_clip)

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
    os.makedirs("asset/sd_reconstruct", exist_ok=True)
    
    # 反归一化原图
    x_denorm = x * imagenet_std + imagenet_mean
    x_denorm = x_denorm.clamp(0, 1)
    
    # 将原图和重建图交替排列：(原图1，重建1，原图2，重建2，...)
    batch_size = x_denorm.shape[0]
    combined = torch.zeros(batch_size * 2, *x_denorm.shape[1:], device=x_denorm.device, dtype=x_denorm.dtype)
    combined[0::2] = x_denorm  # 偶数索引位置放原图
    combined[1::2] = samples   # 奇数索引位置放重建图
    
    # 保存合并后的图像，每行4列
    combined_path = f"asset/sd_reconstruct/combined_{args.step}.png"
    vutils.save_image(combined, combined_path, nrow=4, normalize=False)
    print(f"Combined images (original + reconstructed) saved to {combined_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    reconstruct(args)