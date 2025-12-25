import os
import random
from torch.utils.data import DataLoader
import torch


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
    from datasets import concatenate_datasets
    from functools import partial
    from datasets import load_dataset

    def load_metadata_map(jsonl_path):
        meta_map = {}
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                file_id = os.path.splitext(os.path.basename(item['output_image']))[0]
                meta_map[file_id] = item['instruction']
        return meta_map

    def add_instruction(example, meta_map):
        key = example["__key__"]
        if key in meta_map:
            example["txt"] = meta_map[key]
        else:
            example["txt"] = "" 
        return example

    BLIP3o_60k_data_files = glob.glob(os.path.join(config.BLIP3o_60k_path, "*.tar"))
    BLIP3o_60k_dataset = load_dataset("webdataset", data_files=BLIP3o_60k_data_files, cache_dir=config.BLIP3o_60k_path, split="train", num_proc=32)

    echo4o_instruction_files = glob.glob(os.path.join(config.echo4o_instruction_path, "*.tar.gz"))
    echo4o_instruction_dataset = load_dataset("webdataset", data_files=echo4o_instruction_files, cache_dir=config.echo4o_instruction_path, split="train", num_proc=32)

    echo4o_fantacy_files = glob.glob(os.path.join(config.echo4o_fantacy_path, "*.tar.gz"))
    echo4o_fantacy_dataset = load_dataset("webdataset", data_files=echo4o_fantacy_files, cache_dir=config.echo4o_fantacy_path, split="train", num_proc=32)

    echo4o_instruction_meta_map = load_metadata_map(config.echo4o_instruction_jsonl)
    echo4o_fantacy_meta_map = load_metadata_map(config.echo4o_fantacy_jsonl)

    echo4o_instruction_dataset = echo4o_instruction_dataset.map(
        partial(add_instruction, meta_map=echo4o_instruction_meta_map),
        writer_batch_size=100,  # 减小 batch size 避免 offset overflow
        num_proc=1,  # 使用单进程避免多进程问题
        load_from_cache_file=False  # 禁用缓存，避免缓存问题
    )
    echo4o_fantacy_dataset = echo4o_fantacy_dataset.map(
        partial(add_instruction, meta_map=echo4o_fantacy_meta_map),
        writer_batch_size=100,  # 减小 batch size 避免 offset overflow
        num_proc=1,  # 使用单进程避免多进程问题
        load_from_cache_file=False  # 禁用缓存，避免缓存问题
    )

    combined_dataset = concatenate_datasets([BLIP3o_60k_dataset, echo4o_instruction_dataset, echo4o_fantacy_dataset])
    print(f"BLIP3o_60k_dataset size: {len(BLIP3o_60k_dataset)}")
    print(f"echo4o_instruction size: {len(echo4o_instruction_dataset)}")
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