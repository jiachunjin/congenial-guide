import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch

@torch.inference_mode()
def reconstruct(args):
    from PIL import Image
    from tqdm import trange
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from util.misc import disable_torch_init
    from util.dataloader_llava import load_image

    IMG_START_TOKEN = "<img>"
    IMG_END_TOKEN = "</img>"
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

    exp_dir = args.exp_dir
    exp_name = args.exp_dir.split("/")[-1]
    step = args.step

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    config = OmegaConf.load(os.path.join(exp_dir, f"config.yaml"))

    from model.internvl.modeling_internvl_chat import InternVLChatModel
    from runner.mixture_modality.moe import modify_internvl_to_mixture
    from model.quantizer.multi_vq import get_multi_vq_quantizer

    disable_torch_init()

    internvl = InternVLChatModel.from_pretrained(config.model.internvl_path)
    internvl = modify_internvl_to_mixture(internvl, config.model)
    ckpt_path = os.path.join(exp_dir, f"checkpoint-{step}/pytorch_model/mp_rank_00_model_states.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")["module"]

    m, u = internvl.load_state_dict(ckpt, strict=False)
    print(f"unused keys: {u}")
    internvl = internvl.to(device, dtype).eval()

    quantizer = get_multi_vq_quantizer(config.model.quantizer)
    ckpt_path = config.model.quantizer.ckpt_path
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    quantizer.load_state_dict(ckpt, strict=True)
    quantizer = quantizer.to(device, dtype).eval()

    origin_img_path = args.origin_img_path
    origin_img = Image.open(origin_img_path)
    pixel_value = load_image(origin_img, max_num=12).to(device, dtype)
    vit_feature = internvl.get_vit_feature(pixel_value)
    print(f"pixel_value.shape: {pixel_value.shape}, vit_feature.shape: {vit_feature.shape}")

    prompt = "Reconstruct this image: " + IMG_START_TOKEN + IMG_CONTEXT_TOKEN * 256 + IMG_END_TOKEN + IMG_START_TOKEN
    batch_prompts = [prompt] * args.batch_size
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.internvl_path, trust_remote_code=True, use_fast=False)
    tokenizer_output = tokenizer(
        batch_prompts,
        padding        = True,
        padding_side   = "left",
        truncation     = True,
        return_tensors = "pt",
    )

    input_ids = torch.LongTensor(tokenizer_output["input_ids"]).to(device)
    text_embedding = internvl.language_model.get_input_embeddings()(input_ids)
    print(f"text_embedding.shape: {text_embedding.shape}")

    B, N, C = text_embedding.shape
    text_embedding = text_embedding.reshape(B * N, C)

    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    input_ids = input_ids.reshape(B * N)
    selected = (input_ids == img_context_token_id)
    assert selected.sum() != 0
    text_embedding[selected] = vit_feature.reshape(-1, C).to(device)
    text_embedding = text_embedding.reshape(B, N, C)

    cfg_scale = args.cfg_scale
    tau = 0.9
    topk = 50
    topp = 0.95
    sampling_kwargs = {
        "temperature": tau,
        "top_k": topk,
        "top_p": topp,
        "sample_logits": True
    }

    if cfg_scale > 1:
        # 创建无条件输入：将文本部分替换为pad token
        uncond_input_ids = input_ids.clone()
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        # 找到IMG_START_TOKEN的位置，将其之前的所有token替换为pad_token_id
        img_token_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
        for b in range(uncond_input_ids.shape[0]):
            for t in range(uncond_input_ids.shape[1]):
                if uncond_input_ids[b, t] == img_token_id:
                    break
                uncond_input_ids[b, t] = pad_token_id
        
        uncond_text_embedding = internvl.language_model.get_input_embeddings()(uncond_input_ids)
        text_embedding_cfg = torch.cat([text_embedding, uncond_text_embedding], dim=0)  # (2*B, L, D)
    else:
        text_embedding_cfg = text_embedding

    past_key_values = None
    generated_codes = []
    x_vq_list = []

    for i in trange(256):
        if i == 0:
            current_input = text_embedding_cfg
        else:
            if cfg_scale > 1:
                current_input = torch.cat([img_embeds_current, img_embeds_current], dim=0)
            else:
                current_input = img_embeds_current

        if i == 0:
            vision_token_mask = torch.zeros(current_input.shape[0], current_input.shape[1] - 1, device=device, dtype=dtype)
            vision_token_mask = torch.cat([vision_token_mask, torch.ones(current_input.shape[0], 1, device=device, dtype=dtype)], dim=1)
        else:
            vision_token_mask = torch.ones(current_input.shape[0], current_input.shape[1], device=device, dtype=dtype)

        outputs = internvl.language_model.model(
            inputs_embeds     = current_input,
            use_cache         = True,
            past_key_values   = past_key_values,
            vision_token_mask = vision_token_mask,
        )
        base_token = outputs.last_hidden_state[:, -1:, :]  # (B or 2*B, 1, D)
        past_key_values = outputs.past_key_values

        # generate_from_base_token 返回 (B, K)，无论是否使用 CFG
        generated_code = internvl.ar_head.generate_from_base_token(base_token, cfg_scale, sampling_kwargs)
        z_q_current, x_vq_current = quantizer.indices_to_feature(generated_code.unsqueeze(1))
        img_embeds_current = internvl.visual_projector(z_q_current)  # (B, 1, llm_hidden_size)
        x_vq_list.append(x_vq_current)
        generated_codes.append(generated_code)

    generated_code = torch.stack(generated_codes, dim=1).cpu() # (B, L, K)
    print(generated_code.shape)

    torch.save(generated_code, "./rec_code.pt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--origin_img_path", type=str, required=True)
    args = parser.parse_args()

    reconstruct(args)