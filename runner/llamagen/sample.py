import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import numpy as np
import PIL

@torch.inference_mode()
def generate(
    internvl,
    tokenizer,
    prompt: str,
    temperature: float = 1,
    parallel_size: int = 4,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    device = torch.device("cuda:7"),
):
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).to(device)
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = tokenizer.pad_token_id

    inputs_embeds = internvl.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(device)

    for i in range(image_token_num_per_image):
        outputs = internvl.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
        hidden_states = outputs.last_hidden_state
        
        logits = internvl.visual_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        
        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        # img_embeds = internvl.prepare_gen_img_embeds(next_token)
        img_embeds = internvl.visual_aligner(internvl.visual_embeddings(next_token))
        inputs_embeds = img_embeds.unsqueeze(dim=1)


    dec = internvl.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    os.makedirs('internvl_samples', exist_ok=True)
    for i in range(parallel_size):
        save_path = os.path.join('internvl_samples', "img_{}.jpg".format(i))
        PIL.Image.fromarray(visual_img[i]).save(save_path)

@torch.inference_mode()
def sample_384():
    import os
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer

    from model.llamagen.tokenizer import VQModel, ModelArgs
    from model.internvl.modeling_internvl_chat import InternVLChatModel
    from runner.llamagen.direct_train import modify_internvl

    device = torch.device("cuda:7")
    dtype = torch.bfloat16

    exp_dir = "/inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/experiment/llamagen/1126_intern1B_384/"
    config = OmegaConf.load(os.path.join(exp_dir, "config.yaml"))

    internvl = InternVLChatModel.from_pretrained(config.model.internvl_path)
    internvl = modify_internvl(internvl, config.model.quantizer)
    ckpt_path = os.path.join(exp_dir, "model-llamagen-5000")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    internvl.load_state_dict(ckpt, strict=True)

    internvl = internvl.to(device=device, dtype=dtype).eval()
    tokenizer = AutoTokenizer.from_pretrained(config.model.internvl_path, trust_remote_code=True, use_fast=False)

    tok = VQModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4]))
    tok_ckpt = torch.load(config.model.quantizer.ckpt_path, map_location="cpu", weights_only=True)["model"]
    tok.load_state_dict(tok_ckpt, strict=True)
    tok = tok.to(device, dtype).eval()
    internvl.gen_vision_model = tok

    IMG_START_TOKEN = "<img>"
    prompt_text = "A stunning princess from kabul in red, white traditional clothing, blue eyes, brown hair."

    prompt = f"{prompt_text}" + IMG_START_TOKEN

    generate(internvl, tokenizer, prompt, device=device)


if __name__ == "__main__":
    sample_384()