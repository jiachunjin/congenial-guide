import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
import numpy as np
import PIL

@torch.inference_mode()
def generate(
    mmgpt,
    vl_chat_processor,
    prompt: str,
    temperature: float = 1,
    parallel_size: int = 4,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    device = torch.device("cuda:7"),
):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).to(device)
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(device)

    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
        hidden_states = outputs.last_hidden_state
        
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        
        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)


    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    os.makedirs('generated_samples', exist_ok=True)
    for i in range(parallel_size):
        save_path = os.path.join('generated_samples', "img_{}.jpg".format(i))
        PIL.Image.fromarray(visual_img[i]).save(save_path)

@torch.inference_mode()
def sample_384():
    import os
    from omegaconf import OmegaConf

    from model.janus.models import MultiModalityCausalLM, VLChatProcessor

    device = torch.device("cuda:7")
    dtype = torch.bfloat16

    exp_dir = "/inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/experiment/llamagen/1126_janus_384"
    config = OmegaConf.load(os.path.join(exp_dir, "config.yaml"))

    janus = MultiModalityCausalLM.from_pretrained(config.model.pretrained_path)
    ckpt_path = os.path.join(exp_dir, "model-llamagen-2000")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    janus.load_state_dict(ckpt, strict=True)
    
    janus = janus.to(device=device, dtype=dtype).eval()
    vl_chat_processor = VLChatProcessor.from_pretrained(config.model.pretrained_path)
    tokenizer = vl_chat_processor.tokenizer
    
    prompt_text = "A cute dog playing with a cat."
    prompt = f"Generate an image: {prompt_text}" + vl_chat_processor.image_start_tag
    print(prompt)

    generate(janus, vl_chat_processor, prompt)


if __name__ == "__main__":
    sample_384()