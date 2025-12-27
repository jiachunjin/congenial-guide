import os
import random
from torch.utils.data import DataLoader
import torch

def get_blip3o_validation_dataloader(config, tokenizer):
    journeydb_path = "/inspire/hdd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/jjc/dataset/BLIP3o/BLIP3o-Pretrain-JourneyDB"
    long_path = "/inspire/hdd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/jjc/dataset/BLIP3o/BLIP3o-Pretrain-Long-Caption"
    short_path = "/inspire/hdd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/jjc/dataset/BLIP3o/BLIP3o-Pretrain-Short-Caption"

    journeydb_tars = [
        os.path.join(journeydb_path, "JourneyDB_054.tar"),
        os.path.join(journeydb_path, "JourneyDB_323.tar"),
    ]

    long_tars = [
        os.path.join(long_path, "webdataset_shard_009.tar"),
        os.path.join(long_path, "webdataset_shard_254.tar"),
        os.path.join(long_path, "webdataset_shard_1520.tar"),
        os.path.join(long_path, "sa_000142.tar"),
        os.path.join(long_path, "sa_000623.tar"),
        os.path.join(long_path, "sa_000964.tar"),
        os.path.join(long_path, "sa_000998.tar"),
    ]
    short_tars = [
        os.path.join(short_path, "00214.tar"),
        os.path.join(short_path, "00562.tar"),
    ]
    
    # def count_images_in_tar(tar_path):
    #     """统计tar文件中的图片数量（jpg/png）"""
    #     if not os.path.exists(tar_path):
    #         return 0
    #     count = 0
    #     try:
    #         with tarfile.open(tar_path, 'r') as tar:
    #             for member in tar.getmembers():
    #                 name = member.name.lower()
    #                 if name.endswith('.jpg') or name.endswith('.jpeg') or name.endswith('.png'):
    #                     count += 1
    #     except Exception as e:
    #         print(f"Error reading {tar_path}: {e}")
    #         return 0
    #     return count
    
    # print("=" * 60)
    # print("统计各个tar文件中的图片数量:")
    # print("=" * 60)
    
    # print("\nJourneyDB tars:")
    # journeydb_total = 0
    # for tar_path in journeydb_tars:
    #     count = count_images_in_tar(tar_path)
    #     journeydb_total += count
    #     print(f"  {os.path.basename(tar_path)}: {count} 张图片")
    # print(f"  JourneyDB 总计: {journeydb_total} 张图片")
    
    # print("\nLong tars:")
    # long_total = 0
    # for tar_path in long_tars:
    #     count = count_images_in_tar(tar_path)
    #     long_total += count
    #     print(f"  {os.path.basename(tar_path)}: {count} 张图片")
    # print(f"  Long 总计: {long_total} 张图片")
    
    # print("\nShort tars:")
    # short_total = 0
    # for tar_path in short_tars:
    #     count = count_images_in_tar(tar_path)
    #     short_total += count
    #     print(f"  {os.path.basename(tar_path)}: {count} 张图片")
    # print(f"  Short 总计: {short_total} 张图片")
    
    # print("\n" + "=" * 60)
    # print(f"全部总计: {journeydb_total + long_total + short_total} 张图片")
    # print("=" * 60)

    # 合并所有tar文件
    all_tar_files = journeydb_tars + long_tars + short_tars
    
    # 使用load_dataset加载webdataset格式的数据
    import torchvision.transforms as pth_transforms
    from datasets import load_dataset
    
    # 创建缓存目录（使用第一个tar文件所在目录的父目录）
    cache_dir = os.path.dirname(journeydb_path)
    
    validation_dataset = load_dataset(
        "webdataset", 
        data_files=all_tar_files, 
        cache_dir=cache_dir, 
        split="train", 
        num_proc=32
    )
    
    print(f"Validation dataset size: {len(validation_dataset)}")
    
    # 图像预处理
    preprocess_gen = pth_transforms.Compose([
        pth_transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        pth_transforms.Resize(config.img_size, max_size=None),
        pth_transforms.CenterCrop(config.img_size),
        pth_transforms.ToTensor(),
    ])

    def preprocess_image(image):
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
            max_length     = config.max_seq_length - config.num_img_token,
        )
        input_ids = tokenizer_output["input_ids"]
        if random.random() < config.cfg_drop_rate:
            input_ids[:, 1:-1] = tokenizer.pad_token_id
        attention_mask = tokenizer_output["attention_mask"]

        return input_ids, attention_mask

    def collate_fn(batch):
        pixel_values = []
        input_ids_list = []
        attention_mask_list = []

        for sample in batch:
            pixel_value = preprocess_image(sample["jpg"])
            pixel_values.append(pixel_value)

            text = sample["txt"]
            input_ids, attention_mask = preprocess_text(text)
            input_ids_list.append(input_ids[0])
            attention_mask_list.append(attention_mask[0])
        
        pixel_values = torch.stack(pixel_values)
        input_ids = torch.stack(input_ids_list)
        attention_mask = torch.stack(attention_mask_list)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    # 创建validation dataloader（通常validation不需要shuffle，drop_last设为False）
    dataloader = DataLoader(
        validation_dataset,
        batch_size  = config.val_batch_size,
        shuffle     = False,  # validation通常不打乱
        num_workers = config.num_workers,
        pin_memory  = True,
        drop_last   = False,  # validation通常不丢弃最后一个batch
        collate_fn  = collate_fn,
    )

    return dataloader


def get_blip3o_dataloader(config, tokenizer, accelerator):
    import glob
    import webdataset as wds
    import torchvision.transforms as pth_transforms
    
    urls = []
    for path in config.wds_path:
        urls.extend(glob.glob(os.path.join(path, "*.tar")))

    # 打乱 urls，但不使用固定 seed，让每个进程有不同的随机性
    random.shuffle(urls)

    rank = accelerator.state.process_index
    world_size = accelerator.state.num_processes
    print(f"[Rank {rank}/{world_size}] Found tar files: {len(urls)}")

    preprocess_gen = pth_transforms.Compose([
        pth_transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        pth_transforms.Resize(config.img_size, max_size=None),
        pth_transforms.CenterCrop(config.img_size),
        pth_transforms.ToTensor(),
    ])

    def preprocess_image(image):
        width, height = image.size
        max_size = max(width, height)
        if max_size < config.img_size * 0.75:
            return None
        pixel_values = preprocess_gen(image)

        return pixel_values
    
    def preprocess_text(text):
        IMG_START_TOKEN = "<img>"

        # if random.random() < config.cfg_drop_rate:
        #     text = ""

        prompt = text + IMG_START_TOKEN

        tokenizer_output = tokenizer(
            prompt,
            return_tensors = "pt",
            padding        = "max_length",
            padding_side   = "left",
            truncation     = True,
            max_length     = config.max_seq_length - config.num_img_token,
        )
        input_ids = tokenizer_output["input_ids"]
        if random.random() < config.cfg_drop_rate:
            input_ids[:, 1:-1] = tokenizer.pad_token_id
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

        pixel_values = torch.stack(pixel_values)
        input_ids = torch.stack(input_ids_list)
        attention_mask = torch.stack(attention_mask_list)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    # 使用 ResampledShards: 每个进程独立随机采样 shard，不再静态分配
    # 这样每个进程都能看到所有数据集的数据，且每次采样都不同
    dataset = wds.DataPipeline(
        wds.ResampledShards(urls, seed=rank),  # 每个rank用不同seed，无限采样
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=wds.warn_and_continue), 
        wds.shuffle(bufsize=config.buffer_size, initial=config.buffer_size),
        wds.decode("pil", handler=wds.ignore_and_continue),
        wds.to_tuple("jpg", "txt"),
        wds.map_tuple(preprocess_image, preprocess_text)
    )
    dataloader = DataLoader(
        dataset,
        batch_size         = config.batch_size,
        num_workers        = config.num_workers,
        pin_memory         = True,
        collate_fn         = collation_fn,
        drop_last          = True,
        prefetch_factor    = 4,              # 预取更多 batch
        persistent_workers = True if config.num_workers > 0 else False,  # 保持 worker 进程
    )

    return dataloader


def get_blip3o_dataloader_janus(config, preprocessor, accelerator):
    import glob
    import webdataset as wds
    import torchvision.transforms as pth_transforms
    
    urls = []
    for path in config.wds_path:
        urls.extend(glob.glob(os.path.join(path, "*.tar")))

    random.shuffle(urls)

    rank = accelerator.state.process_index
    world_size = accelerator.state.num_processes
    print(f"[Rank {rank}/{world_size}] Found tar files: {len(urls)}")

    preprocess_gen = pth_transforms.Compose([
        pth_transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        pth_transforms.Resize(config.img_size, max_size=None),
        pth_transforms.CenterCrop(config.img_size),
        pth_transforms.ToTensor(),
    ])

    def preprocess_image(image):
        width, height = image.size
        max_size = max(width, height)
        if max_size < config.img_size * 0.75:
            return None
        pixel_values = preprocess_gen(image)

        return pixel_values
    
    def preprocess_text(text):
        if random.random() < config.cfg_drop_rate:
            text = ""

        prompt = f"Generate an image: {text}" + preprocessor.image_start_tag

        tokenizer_output = preprocessor.tokenizer(
            prompt,
            return_tensors = "pt",
            padding        = "max_length",
            padding_side   = "left",
            truncation     = True,
            max_length     = config.max_seq_length - config.num_img_token,
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

        pixel_values = torch.stack(pixel_values)
        input_ids = torch.stack(input_ids_list)
        attention_mask = torch.stack(attention_mask_list)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    # 使用 ResampledShards: 每个进程独立随机采样 shard，不再静态分配
    dataset = wds.DataPipeline(
        wds.ResampledShards(urls, seed=rank),  # 每个rank用不同seed，无限采样
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=wds.warn_and_continue), 
        wds.shuffle(bufsize=config.buffer_size, initial=config.buffer_size),
        wds.decode("pil", handler=wds.ignore_and_continue),
        wds.to_tuple("jpg", "txt"),
        wds.map_tuple(preprocess_image, preprocess_text)
    )
    dataloader = DataLoader(
        dataset,
        batch_size         = config.batch_size,
        num_workers        = config.num_workers,
        pin_memory         = True,
        collate_fn         = collation_fn,
        drop_last          = True,
        prefetch_factor    = 4,              # 预取更多 batch
        persistent_workers = True if config.num_workers > 0 else False,  # 保持 worker 进程
    )

    return dataloader

def get_blip3o_60k_dataloader(config, tokenizer):
    import os
    import glob
    import torchvision.transforms as pth_transforms
    from datasets import load_dataset

    data_files = glob.glob(os.path.join(config.wds_path, "*.tar"))
    BLIP3o_60k_dataset = load_dataset("webdataset", data_files=data_files, cache_dir=config.cache_dir, split="train", num_proc=32)

    preprocess_gen = pth_transforms.Compose([
        pth_transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        pth_transforms.Resize(config.img_size, max_size=None),
        pth_transforms.CenterCrop(config.img_size),
        pth_transforms.ToTensor(),
    ])

    def preprocess_image(image):
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
            max_length     = config.max_seq_length - config.num_img_token,
        )
        input_ids = tokenizer_output["input_ids"]
        if random.random() < config.cfg_drop_rate:
            input_ids[:, 1:-1] = tokenizer.pad_token_id
        attention_mask = tokenizer_output["attention_mask"]

        return input_ids, attention_mask

    def collate_fn(batch):
        pixel_values = []
        input_ids_list = []
        attention_mask_list = []

        for sample in batch:
            pixel_value = preprocess_image(sample["jpg"])
            pixel_values.append(pixel_value)

            text = sample["txt"]
            input_ids, attention_mask = preprocess_text(text)
            input_ids_list.append(input_ids[0])
            attention_mask_list.append(attention_mask[0])
        
        pixel_values = torch.stack(pixel_values)
        input_ids = torch.stack(input_ids_list)
        attention_mask = torch.stack(attention_mask_list)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    dataloader = torch.utils.data.DataLoader(
        BLIP3o_60k_dataset,
        batch_size  = config.batch_size,
        shuffle     = True,
        num_workers = config.num_workers,
        pin_memory  = True,
        drop_last   = True,
        collate_fn  = collate_fn,
    )

    return dataloader

def get_blip3o_echo_4o_dataloader(config, tokenizer):
    import os
    import json
    import glob
    import torchvision.transforms as pth_transforms
    from datasets import concatenate_datasets, load_dataset

    use_echo_fantacy = "echo4o_fantacy_path" in config

    def load_metadata_map(jsonl_path):
        meta_map = {}
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                file_id = os.path.splitext(os.path.basename(item['output_image']))[0]
                meta_map[file_id] = item['instruction']
        return meta_map

    BLIP3o_60k_data_files = glob.glob(os.path.join(config.BLIP3o_60k_path, "*.tar"))
    BLIP3o_60k_dataset = load_dataset("webdataset", data_files=BLIP3o_60k_data_files, cache_dir=config.BLIP3o_60k_path, split="train", num_proc=32)

    echo4o_instruction_files = glob.glob(os.path.join(config.echo4o_instruction_path, "*.tar.gz"))
    echo4o_instruction_dataset = load_dataset("webdataset", data_files=echo4o_instruction_files, cache_dir=config.echo4o_instruction_path, split="train", num_proc=32)

    if use_echo_fantacy:
        echo4o_fantacy_files = glob.glob(os.path.join(config.echo4o_fantacy_path, "*.tar.gz"))
        echo4o_fantacy_dataset = load_dataset("webdataset", data_files=echo4o_fantacy_files, cache_dir=config.echo4o_fantacy_path, split="train", num_proc=32)
        echo4o_fantacy_meta_map = load_metadata_map(config.echo4o_fantacy_jsonl)
        echo4o_fantacy_dataset = echo4o_fantacy_dataset.add_column("_source", ["fantacy"] * len(echo4o_fantacy_dataset))

    # 加载两个独立的 meta_map（key 可能重复，不能合并）
    echo4o_instruction_meta_map = load_metadata_map(config.echo4o_instruction_jsonl)

    # 给数据集添加来源标记，用于在 collate_fn 中区分（add_column 比 .map() 快很多）
    BLIP3o_60k_dataset = BLIP3o_60k_dataset.add_column("_source", ["blip3o"] * len(BLIP3o_60k_dataset))
    echo4o_instruction_dataset = echo4o_instruction_dataset.add_column("_source", ["instruction"] * len(echo4o_instruction_dataset))

    if use_echo_fantacy:
        combined_dataset = concatenate_datasets([BLIP3o_60k_dataset, echo4o_instruction_dataset, echo4o_fantacy_dataset])
    else:
        combined_dataset = concatenate_datasets([BLIP3o_60k_dataset, echo4o_instruction_dataset])

    print(f"BLIP3o_60k_dataset size: {len(BLIP3o_60k_dataset)}")
    print(f"echo4o_instruction size: {len(echo4o_instruction_dataset)}")
    if use_echo_fantacy:
        print(f"echo4o_fantacy size: {len(echo4o_fantacy_dataset)}")
    print(f"combined_dataset size: {len(combined_dataset)}")

    preprocess_gen = pth_transforms.Compose([
        pth_transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        pth_transforms.Resize(config.img_size, max_size=None),
        pth_transforms.CenterCrop(config.img_size),
        pth_transforms.ToTensor(),
    ])

    def preprocess_image(image):
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
            max_length     = config.max_seq_length - config.num_img_token,
        )
        input_ids = tokenizer_output["input_ids"]
        if random.random() < config.cfg_drop_rate:
            input_ids[:, 1:-1] = tokenizer.pad_token_id
        attention_mask = tokenizer_output["attention_mask"]

        return input_ids, attention_mask

    def collate_fn(batch):
        pixel_values = []
        input_ids_list = []
        attention_mask_list = []

        for sample in batch:
            pixel_value = preprocess_image(sample["jpg"])
            pixel_values.append(pixel_value)

            # 根据来源选择对应的 meta_map 动态查找 instruction
            source = sample.get("_source")
            key = sample.get("__key__")
            if source == "instruction" and key in echo4o_instruction_meta_map:
                text = echo4o_instruction_meta_map[key]
            elif source == "fantacy" and key in echo4o_fantacy_meta_map:
                text = echo4o_fantacy_meta_map[key]
            else:
                text = sample.get("txt", "")
            
            input_ids, attention_mask = preprocess_text(text)
            input_ids_list.append(input_ids[0])
            attention_mask_list.append(attention_mask[0])
        
        pixel_values = torch.stack(pixel_values)
        input_ids = torch.stack(input_ids_list)
        attention_mask = torch.stack(attention_mask_list)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    dataloader = DataLoader(
        combined_dataset,
        batch_size  = config.batch_size,
        shuffle     = True,
        num_workers = config.num_workers,
        pin_memory  = True,
        drop_last   = True,
        collate_fn  = collate_fn,
    )

    return dataloader

if __name__ == "__main__":
    from transformers import AutoTokenizer
    from omegaconf import OmegaConf
    config = OmegaConf.load("config/sft/echo4o_blip3o.yaml")
    config.data.batch_size = 2
    tokenizer = AutoTokenizer.from_pretrained(config.model.internvl_path, trust_remote_code=True, use_fast=False)

    dataloader = get_blip3o_validation_dataloader(config.data, tokenizer)
    for batch in dataloader:
        print(batch)
        break