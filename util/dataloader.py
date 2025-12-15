import os
import random
from torch.utils.data import DataLoader
import torch
import itertools


def get_blip3o_dataloader(config, tokenizer, accelerator):
    import glob
    import webdataset as wds
    import torchvision.transforms as pth_transforms
    
    urls = []
    for path in config.wds_path:
        urls.extend(glob.glob(os.path.join(path, "*.tar")))

    print(f"Found tar files: {len(urls)}")

    def nodesplitter(src, group=None):
        if accelerator.state.num_processes > 1:
            rank = accelerator.state.process_index
            world_size = accelerator.state.num_processes

            return itertools.islice(src, rank, None, world_size)
        return src

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

    dataset = wds.DataPipeline(
        wds.SimpleShardList(urls),
        nodesplitter,
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=wds.warn_and_continue), 
        wds.shuffle(config.buffer_size),
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

    print(f"Found tar files: {len(urls)}")

    def nodesplitter(src, group=None):
        if accelerator.state.num_processes > 1:
            rank = accelerator.state.process_index
            world_size = accelerator.state.num_processes

            return itertools.islice(src, rank, None, world_size)
        return src

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

    dataset = wds.DataPipeline(
        wds.SimpleShardList(urls),
        nodesplitter,
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=wds.warn_and_continue), 
        wds.shuffle(config.buffer_size),
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