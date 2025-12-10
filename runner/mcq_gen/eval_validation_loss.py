import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from einops import rearrange
from tqdm import tqdm

@torch.inference_mode()
def eval_validation_loss(args):
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    import webdataset as wds
    import torchvision.transforms as pth_transforms
    from model.internvl.modeling_internvl_chat import InternVLChatModel
    from runner.mcq_gen.dev_my_ar_head import modify_internvl_my_ar_head
    from model.quantizer.multi_vq import get_multi_vq_quantizer

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    exp_dir = args.exp_dir
    step = args.step
    config = OmegaConf.load(os.path.join(exp_dir, f"config.yaml"))
    
    val_data_tar = "/inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/dataset/BLIP3o/BLIP3o-Pretrain-Short-Caption/00562.tar"

    # 加载模型
    internvl = InternVLChatModel.from_pretrained(config.model.internvl_path)
    internvl = modify_internvl_my_ar_head(internvl, config.model)
    ckpt_path = os.path.join(exp_dir, f"model-mcq_gen-{step}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    internvl.load_state_dict(ckpt, strict=True)
    internvl = internvl.to(device, dtype).eval()
    print("load internvl done")

    quantizer = get_multi_vq_quantizer(config.model.quantizer)
    ckpt_path = config.model.quantizer.ckpt_path
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    quantizer.load_state_dict(ckpt, strict=True)
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
        batch_size  = config.data.batch_size,
        num_workers = config.data.num_workers,
        pin_memory  = True,
        collate_fn  = collation_fn,
        drop_last   = False,
    )

    # 计算validation loss
    total_loss = 0.0
    total_samples = 0
    V = config.model.head.num_embeddings

    print("开始计算validation loss...")
    for batch_idx, batch in enumerate(tqdm(dataloader)):
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
        z_q, code = quantizer.get_zq_indices(vit_feature)  # z_q: (B, L, 256), code: (B, L, K)

        B, L, K = code.shape

        # 处理文本和视觉特征
        text_embedding_t2i = internvl.language_model.get_input_embeddings()(input_ids)
        visual_embedding_t2i = internvl.visual_projector(z_q)
        joint_embedding = torch.cat([text_embedding_t2i, visual_embedding_t2i], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones((B, config.data.num_img_token), dtype=torch.bool, device=device)], dim=1)

        # 获取visual hidden states
        visual_hidden_states = internvl.language_model(
            inputs_embeds        = joint_embedding,
            attention_mask       = attention_mask,
            output_hidden_states = True,
        ).hidden_states[-1][:, -config.data.num_img_token-1:-1, :]  # (B, L, D)

        # 计算logits和loss
        prefix = rearrange(visual_hidden_states, "B L D -> (B L) 1 D")
        head_visual_embeddings = internvl.ar_head._code_to_embeddings(code)  # (BxL, K, D)
        h = torch.cat((prefix, head_visual_embeddings), dim=1)  # (BxL, K+1, D)

        logits = internvl.ar_head(h[:, :-1, :])  # (BxL, K, V)
        logits = rearrange(logits, "(B L) K V -> B L K V", B=B, L=L)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, V), code.view(-1))

        # cross_entropy已经对所有元素求平均，所以loss是batch内所有样本的平均值
        # 为了得到整个数据集的平均loss，需要按样本数加权平均
        num_samples_in_batch = B * L * K
        total_loss += loss.item() * num_samples_in_batch
        total_samples += num_samples_in_batch

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    print(f"Validation CE Loss: {avg_loss:.6f}")
    print(f"Total samples: {total_samples}")

    return avg_loss

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()

    eval_validation_loss(args)