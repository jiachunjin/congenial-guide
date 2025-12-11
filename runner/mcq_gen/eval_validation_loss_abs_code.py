import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from tqdm import tqdm
from util.sample import sample
from einops import rearrange

@torch.inference_mode()
def eval_validation_loss(args):
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    import webdataset as wds
    import torchvision.transforms as pth_transforms
    from model.internvl.modeling_internvl_chat import InternVLChatModel
    from runner.mcq_gen.dev_abs_code import modify_internvl
    from runner.mcq_gen.dev_ar_head import load_quantizer

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    exp_dir = args.exp_dir
    step = args.step
    config = OmegaConf.load(os.path.join(exp_dir, f"config.yaml"))
    
    val_data_tar = "/inspire/hdd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/jjc/dataset/BLIP3o/BLIP3o-Pretrain-Short-Caption/00512.tar"

    # 加载模型
    internvl = InternVLChatModel.from_pretrained(config.model.internvl_path)
    internvl = modify_internvl(internvl, config.model)
    ckpt_path = os.path.join(exp_dir, f"model-mcq_gen-{step}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    internvl.load_state_dict(ckpt, strict=True)
    internvl = internvl.to(device, dtype).eval()
    print("load internvl done")

    quantizer = load_quantizer(config.model.quantizer)
    quantizer = quantizer.to(device, dtype).eval()
    print(f"load quantizer done")

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.internvl_path, trust_remote_code=True, use_fast=False)
    
    # 创建validation dataloader
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    imagenet_mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    imagenet_std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1)

    preprocess_gen = pth_transforms.Compose([
        pth_transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        pth_transforms.Resize(config.data.img_size, max_size=None),
        pth_transforms.CenterCrop(config.data.img_size),
        pth_transforms.ToTensor(),
    ])

    def preprocess_image(image):
        width, height = image.size
        max_size = max(width, height)
        if max_size < config.data.img_size * 0.75:
            return None
        pixel_values = preprocess_gen(image)
        return pixel_values
    
    def preprocess_text(text):
        IMG_START_TOKEN = "<img>"
        prompt = text + IMG_START_TOKEN
        tokenizer_output = tokenizer(
            prompt,
            return_tensors = "pt",
            padding        = "max_length",
            padding_side   = "left",
            truncation     = True,
            max_length     = config.data.max_seq_length - config.data.num_img_token,
        )
        input_ids = tokenizer_output["input_ids"]
        attention_mask = tokenizer_output["attention_mask"]
        return input_ids, attention_mask

    def collation_fn(batch):
        pixel_values = []
        input_ids_list = []
        attention_mask_list = []

        for sample in batch:
            pixel_value, (input_ids, attention_mask) = sample
            if pixel_value == None:
                continue
            else:
                pixel_values.append(pixel_value)
                input_ids_list.append(input_ids[0])
                attention_mask_list.append(attention_mask[0])

        if len(pixel_values) == 0:
            return None

        pixel_values = torch.stack(pixel_values)
        input_ids = torch.stack(input_ids_list)
        attention_mask = torch.stack(attention_mask_list)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    dataset = wds.DataPipeline(
        wds.SimpleShardList([val_data_tar]),
        wds.tarfile_to_samples(handler=wds.warn_and_continue), 
        wds.decode("pil", handler=wds.ignore_and_continue),
        wds.to_tuple("jpg", "txt"),
        wds.map_tuple(preprocess_image, preprocess_text)
    )
    dataloader = DataLoader(
        dataset,
        batch_size  = 20,
        num_workers = config.data.num_workers,
        pin_memory  = True,
        collate_fn  = collation_fn,
        drop_last   = False,
    )

    # 计算validation loss
    total_loss = 0.0
    total_samples = 0
    K = config.model.quantizer.num_codebooks
    V = config.model.quantizer.num_embeddings
    
    # 用于统计熵：按位置索引收集，每个位置收集所有样本的熵
    # position_entropies[i] 存储位置i的所有熵值
    position_entropies = {}  # {position_idx: [entropy_values]}

    print("开始计算validation loss...")
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        if batch_idx > 20: break
        if batch is None:
            continue
        
        pixel_values = batch["pixel_values"].to(device, dtype)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        x_gen = (pixel_values - imagenet_mean) / imagenet_std
        B = x_gen.shape[0]

        # 获取vit feature和code
        internvl.vision_model.eval()
        vit_feature = internvl.get_vit_feature(x_gen)
        _, code, _ = quantizer(vit_feature)  # code: (B, L, K)
        abs_code = quantizer.to_abs_code(code)  # (B, LxK)
        # 处理文本和视觉特征
        text_embedding_t2i = internvl.language_model.get_input_embeddings()(input_ids)
        visual_embedding_t2i = internvl.visual_embeddings(abs_code)
        joint_embedding = torch.cat([text_embedding_t2i, visual_embedding_t2i], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones((B, config.data.num_img_token), dtype=torch.bool, device=device)], dim=1)

        # 获取visual hidden states
        visual_hidden_states = internvl.language_model(
            inputs_embeds        = joint_embedding,
            attention_mask       = attention_mask,
            output_hidden_states = True,
        ).hidden_states[-1][:, -config.data.num_img_token-1:-1, :]  # (B, L, D)

        # 计算logits和loss
        logits = internvl.head(visual_hidden_states) # (B, 2048， 16384)
        
        # 从logits中采样得到code
        # reshape logits to (B*L*K, K*V) for sampling
        # logits_flat = logits.view(-1, K*V)  # (B*L*K, K*V)
        
        # 采样参数（参考 gen_abs_code.py）
        tau = 1
        topk = 2048
        topp = 1
        sampling_kwargs = {
            "temperature": tau,
            "top_k": topk,
            "top_p": topp,
            "sample_logits": False
        }
        
        B, L, V = logits.shape
        logits = rearrange(logits, "B L V -> (B L) V")
        sampled_tokens, _ = sample(logits, **sampling_kwargs)  # (B*L, 1)
        abs_code_sampled = sampled_tokens.squeeze(-1)  # (B*L,)
        abs_code_sampled = rearrange(abs_code_sampled, "(B L) -> B L", B=B, L=L) # (B, 2048,)
        print("abs_code_sampled", abs_code_sampled.shape)
        print(abs_code_sampled.shape, abs_code.shape)
        accuracy = (abs_code_sampled == abs_code).sum().item() / abs_code_sampled.numel()
        print(f"Accuracy: {accuracy:.6f}")
        
        # 将abs_code转换为rel_code
        rel_code_sampled = quantizer.to_rel_code(abs_code_sampled)  # (B, L, K)
        print("rel_code_sampled", rel_code_sampled.shape)
        torch.save(rel_code_sampled, f"asset/mcq_gen/val_rel_code_sampled_{step}.pt")
        loss = torch.nn.functional.cross_entropy(logits, abs_code.view(-1))
        print(f"Loss: {loss.item():.6f}")

        torch.save(quantizer.to_rel_code(abs_code), f"asset/mcq_gen/val_rel_code_GT_{step}.pt")
        accuracy = (abs_code_sampled == abs_code).sum().item() / abs_code_sampled.numel()
        print(f"Accuracy: {accuracy:.6f}")

        exit(0)
    
    #     # 计算每个位置的熵
    #     probs = torch.nn.functional.softmax(logits, dim=-1)  # (B, LxK, K*V)
    #     log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # (B, LxK, K*V)
    #     entropies = -torch.sum(probs * log_probs, dim=-1)  # (B, LxK)
        
    #     # 直接按位置索引收集熵值（保持L*K个位置）
    #     entropies_cpu = entropies.cpu().detach()
    #     LxK = entropies_cpu.shape[1]  # L*K
        
    #     for pos_idx in range(LxK):
    #         if pos_idx not in position_entropies:
    #             position_entropies[pos_idx] = []
    #         # 收集该位置所有batch样本的熵值
    #         position_entropies[pos_idx].extend(entropies_cpu[:, pos_idx].tolist())

    #     # cross_entropy已经对所有元素求平均，所以loss是batch内所有样本的平均值
    #     # 为了得到整个数据集的平均loss，需要按样本数加权平均
    #     num_samples_in_batch = B * L * K
    #     total_loss += loss.item() * num_samples_in_batch
    #     total_samples += num_samples_in_batch

    # avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    # print(f"Validation CE Loss: {avg_loss:.6f}")
    # print(f"Total samples: {total_samples}")
    
    # # 统计每个位置的熵并绘制图表
    # if len(position_entropies) > 0:
    #     import numpy as np
    #     import matplotlib.pyplot as plt
        
    #     # 计算每个位置的平均熵
    #     num_positions = max(position_entropies.keys()) + 1
    #     position_mean_entropies = []
        
    #     print(f"\n=== 每个位置的熵统计 ===")
    #     for pos_idx in range(num_positions):
    #         if pos_idx in position_entropies:
    #             pos_entropy_values = np.array(position_entropies[pos_idx])
    #             mean_entropy = np.mean(pos_entropy_values)
    #             position_mean_entropies.append(mean_entropy)
    #         else:
    #             position_mean_entropies.append(0.0)  # 如果某个位置没有数据，设为0
        
    #     position_mean_entropies = np.array(position_mean_entropies)
        
    #     # 打印统计信息
    #     print(f"总位置数: {num_positions}")
    #     print(f"平均熵: {np.mean(position_mean_entropies):.6f}")
    #     print(f"标准差: {np.std(position_mean_entropies):.6f}")
    #     print(f"最小值: {np.min(position_mean_entropies):.6f} (位置 {np.argmin(position_mean_entropies)})")
    #     print(f"最大值: {np.max(position_mean_entropies):.6f} (位置 {np.argmax(position_mean_entropies)})")
        
    #     # 绘制图表（只画前128个位置）
    #     num_plot = min(128, num_positions)
    #     positions = np.arange(num_plot)
    #     plt.figure(figsize=(12, 6))
    #     plt.bar(positions, position_mean_entropies[:num_plot], width=1.0, alpha=0.7)
    #     plt.xlabel('Position Index', fontsize=12)
    #     plt.ylabel('Entropy', fontsize=12)
    #     plt.title(f'Entropy Distribution Across First {num_plot} Visual Token Positions', fontsize=14)
    #     plt.grid(True, alpha=0.3, axis='y')
    #     plt.tight_layout()
        
    #     # 保存图片
    #     output_path = os.path.join(exp_dir, f"entropy_by_position_step_{step}.png")
    #     plt.savefig(output_path, dpi=150, bbox_inches='tight')
    #     print(f"\n熵分布图已保存到: {output_path}")
        
    #     # 也保存数据到文件
    #     data_output_path = os.path.join(exp_dir, f"entropy_by_position_step_{step}.npy")
    #     np.save(data_output_path, position_mean_entropies)
    #     print(f"熵数据已保存到: {data_output_path}")

    return avg_loss

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()

    eval_validation_loss(args)
