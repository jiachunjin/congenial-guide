import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from einops import rearrange
from util.misc import disable_torch_init
from accelerate import Accelerator

@torch.inference_mode()
def intern_gen(internvl, quantizer, tokenizer, prompt, batch_size, cfg_scale, sampling_kwargs, device):
    from tqdm import trange
    dtype = torch.bfloat16
    IMG_START_TOKEN = "<img>"
    prompt = prompt + IMG_START_TOKEN

    batch_prompts = [prompt] * batch_size

    tokenizer_output = tokenizer(
        batch_prompts,
        padding        = True,
        padding_side   = "left",
        truncation     = True,
        return_tensors = "pt",
    )

    input_ids = torch.LongTensor(tokenizer_output["input_ids"]).to(device)
    text_embedding = internvl.language_model.get_input_embeddings()(input_ids)

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
                # 条件和无条件分支使用相同的 img_embeds（因为采样结果相同）
                # 拼接成 (2*B, 1, D) 的形式以匹配 KV cache 的 batch 维度
                current_input = torch.cat([img_embeds_current, img_embeds_current], dim=0)
            else:
                current_input = img_embeds_current

        # 构建 vision_token_mask: 第一次是文本(0)，后续是视觉(1)
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

    generated_code = torch.stack(generated_codes, dim=1) # (B, L, K)
    return generated_code


@torch.inference_mode()
def generate(args):
    from omegaconf import OmegaConf

    accelerator = Accelerator()
    device = accelerator.device
    dtype = torch.bfloat16
    exp_dir = args.exp_dir
    exp_name = args.exp_dir.split("/")[-1]
    step = args.step
    config = OmegaConf.load(os.path.join(exp_dir, f"config.yaml"))

    from model.internvl.modeling_internvl_chat import InternVLChatModel
    from runner.mixture_modality.moe import modify_internvl_to_mixture
    from model.quantizer.multi_vq import get_multi_vq_quantizer

    disable_torch_init()

    internvl = InternVLChatModel.from_pretrained(config.model.internvl_path)
    internvl = modify_internvl_to_mixture(internvl, config.model)
    ckpt_path = os.path.join(exp_dir, f"checkpoint-{step}/pytorch_model/mp_rank_00_model_states.pt")
    # ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # if "model" in ckpt:
    #     ckpt = ckpt["model"]
    # ckpt_path = "/checkpoint-35000/pytorch_model/mp_rank_00_model_states.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")["module"]

    m, u = internvl.load_state_dict(ckpt, strict=False)
    accelerator.print(f"unused keys: {u}")
    internvl = internvl.to(device, dtype).eval()

    quantizer = get_multi_vq_quantizer(config.model.quantizer)
    ckpt_path = config.model.quantizer.ckpt_path
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    quantizer.load_state_dict(ckpt, strict=True)
    quantizer = quantizer.to(device, dtype).eval()

    # ---------- generate from text prompt ----------
    from tqdm import trange
    from transformers import AutoTokenizer

    IMG_START_TOKEN = "<img>"
    tokenizer = AutoTokenizer.from_pretrained(config.model.internvl_path, trust_remote_code=True, use_fast=False)
    if args.rewrite:
        prompts = [
            "A photorealistic still life of a crystal wine glass placed to the right of a grilled hot dog, studio lighting, sharp focus.",
            "A wide-angle shot of four retro television sets arranged in a perfect horizontal line against a plain wall.",
            "A high-quality photo of a professional tennis racket resting next to a glass of red wine on a wooden surface.",
            "A modern living room scene featuring a large TV and a vintage bicycle leaning against the wall.",
            "A dark green blackboard with the words 'Hello, ICML 2026' written clearly in white chalk, centered composition.",
            "A vibrant photo of a purple school backpack and a bright yellow umbrella on a white floor.",
            "A clean white board with the text 'Visual thinking without pixels' written in the center, minimalist style, highly detailed.",
            "A breathtaking portrait of a princess from Kabul in traditional red and white attire, striking blue eyes, brown hair, cinematic light.",
            "A sleek blue smartphone and a crisp green apple on a pure white background, professional product photography.",
            "A top-down view of a mechanical keyboard with a fresh pizza placed on the desk below it.",
            "A minimalist photo of two identical analog clocks mounted side-by-side on a white wall.",
            "A surrealist high-quality photo of a banana with vibrant blue peel, soft shadows, 8k resolution."
        ]
    else:
        prompts = [
            "a photo of a wine glass right of a hot dog",
            "a photo of 4 TVs in a line",
            "a photo of a tennis racket and a wine glass",
            "a photo of a tv and a bicycle",
            "A blackboard with words 'Hello, ICML 2026', no other words exist",
            "A blackboard with words 'In recent years, multimodal learning has attracted significant attention due to its ability to integrate vision, language, and reasoning. Despite impressive progress, generating images that contain long, perfectly readable text remains an open research problem.'",
            "A blackboard with words 'Text-to-image models often fail when rendering long paragraphs. Explicit constraints on layout and typography improve text fidelity. Structured prompts significantly reduce spelling errors.'",
            "A photo of a purple backpack and a yellow unbrella.",
            "A whiteboard with 'Visual thinking without pixels' in the center.",
            "a photo of a red orange and a purple broccoli",
            "a photo of a blue cell phone and a green apple with white background",
            "a photo of a pizza below a computer keyboard",
            "a photo of a blue banana",
            "A soft, natural portrait photograph captures a young woman with fair skin and long, ash-blonde hair cascading gently over her shoulders. At the very bottom of the frame, in simple, elegant lettering, appears the phrase 'BE KIND'",
            "The image depicts a modern, multi-story building with a white facade and numerous balconies. The structure is partially covered in purple scaffolding on the right side, indicating ongoing construction or renovation. The building is situated in an urban area with clear blue skies above. In front of the building, there is a paved plaza with some greenery and a few palm trees. A street lamp stands prominently on the left side of the plaza. To the right, part of another building with a beige exterior is visible. The scene suggests a sunny day in a developed cityscape.",
            "A photo of 4 TVs in a row, with a white background",
            "The image depicts the American Red Cross building, characterized by its neoclassical architectural style. The structure features tall, white columns supporting a pediment and a balustrade at the top. The facade is adorned with large windows, some of which have red crosses, symbolizing the organization's humanitarian mission. The building is set against a clear blue sky, with a tree partially obscuring the right side of the image. The overall appearance suggests a sense of stability and dedication to service, reflecting the Red Cross's commitment to aid and support.",
            "Scientist at Sunway University conducts research in a laboratory setting.",
            "Muscular man in workout attire, standing confidently by a railing.",
            "Confident man in leather jacket leaning against a wall.",
            "A detailed ink illustration of a hedgehog.",
            "A cheetah is sitting in the grass on a grassland at sunset.",
            "An oil portrait: A young woman in a flower field at sunset, with mountains in the background."
        ]
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
    batch_size = args.batch_size
    
    # 将任务分配到不同的GPU
    num_processes = accelerator.num_processes
    process_index = accelerator.process_index
    
    all_generated_codes = []
    all_prompt_indices = []
    
    for idx, prompt_txt in enumerate(prompts):
        # 只处理分配给当前进程的任务
        if idx % num_processes != process_index:
            continue
            
        all_prompt_indices.append(idx)
        prompt = prompt_txt + IMG_START_TOKEN
        accelerator.print(f"Process {process_index}: Processing prompt {idx}/{len(prompts)}: '{prompt_txt[:50]}...'")
        # 将单个 prompt 扩展为 batch_size 个副本
        batch_prompts = [prompt] * batch_size
        tokenizer_output = tokenizer(
            batch_prompts,
            padding        = True,
            padding_side   = "left",
            truncation     = True,
            return_tensors = "pt",
        )

        input_ids = torch.LongTensor(tokenizer_output["input_ids"]).to(device)
        text_embedding = internvl.language_model.get_input_embeddings()(input_ids)

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
                    # 条件和无条件分支使用相同的 img_embeds（因为采样结果相同）
                    # 拼接成 (2*B, 1, D) 的形式以匹配 KV cache 的 batch 维度
                    current_input = torch.cat([img_embeds_current, img_embeds_current], dim=0)
                else:
                    current_input = img_embeds_current

            # 构建 vision_token_mask: 第一次是文本(0)，后续是视觉(1)
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

        generated_code = torch.stack(generated_codes, dim=1) # (B, L, K)
        all_generated_codes.append(generated_code)
        accelerator.print(f"Process {process_index}: Generated code shape: {generated_code.shape}")
    
    # 每个进程将结果保存到临时文件
    os.makedirs("asset/code", exist_ok=True)
    temp_code_path = f"asset/code/{exp_name}_{step}_{cfg_scale}_{tau}_{topk}_{topp}_{args.rewrite}_rank{process_index}.pt"
    temp_indices_path = f"asset/code/{exp_name}_{step}_{cfg_scale}_{tau}_{topk}_{topp}_{args.rewrite}_rank{process_index}_indices.pt"
    
    if all_generated_codes:
        # 保存当前进程的 codes 和对应的 prompt 索引
        # all_generated_codes 是一个列表，每个元素是 (batch_size, L, K) 的 tensor
        # 我们需要保存为字典，键为 prompt 索引，值为对应的 code tensor
        codes_dict = {idx: code for idx, code in zip(all_prompt_indices, all_generated_codes)}
        torch.save(codes_dict, temp_code_path)
        torch.save(torch.tensor(all_prompt_indices), temp_indices_path)
        accelerator.print(f"Process {process_index}: Saved {len(all_generated_codes)} codes to {temp_code_path}")
    
    # 等待所有进程完成
    accelerator.wait_for_everyone()
    
    # 只在主进程（rank 0）进行合并和保存
    if accelerator.is_main_process:
        all_codes_list = []
        
        # 收集所有进程的结果
        for rank in range(num_processes):
            rank_code_path = f"asset/code/{exp_name}_{step}_{cfg_scale}_{tau}_{topk}_{topp}_{args.rewrite}_rank{rank}.pt"
            rank_indices_path = f"asset/code/{exp_name}_{step}_{cfg_scale}_{tau}_{topk}_{topp}_{args.rewrite}_rank{rank}_indices.pt"
            
            if os.path.exists(rank_code_path) and os.path.exists(rank_indices_path):
                rank_codes_dict = torch.load(rank_code_path, map_location="cpu")
                rank_indices = torch.load(rank_indices_path, map_location="cpu").tolist()
                
                # 将每个 code 与其对应的 prompt 索引配对
                for idx in rank_indices:
                    if idx in rank_codes_dict:
                        all_codes_list.append((idx, rank_codes_dict[idx]))
                
                # 删除临时文件
                os.remove(rank_code_path)
                os.remove(rank_indices_path)
        
        if all_codes_list:
            # 按照原始 prompt 索引排序
            all_codes_list.sort(key=lambda x: x[0])
            
            # 提取排序后的 codes
            sorted_codes = [code for _, code in all_codes_list]
            
            # 合并所有 codes
            all_codes = torch.cat(sorted_codes, dim=0).cpu()  # (total_B, L, K) where total_B is total batch size across all prompts
            code_path = f"asset/code/{exp_name}_{step}_{cfg_scale}_{tau}_{topk}_{topp}_{args.rewrite}.pt"
            torch.save(all_codes, code_path)
            accelerator.print(f"All codes saved to {code_path}, shape: {all_codes.shape}")
        else:
            accelerator.print("Warning: No codes generated!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--rewrite", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    
    args = parser.parse_args()

    generate(args)