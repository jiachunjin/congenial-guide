import os
import json
import math
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from util.dataloader_llava import load_image
# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder):
        self.questions = questions
        self.image_folder = image_folder

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        pixel_values = load_image(image, max_num=12)
        return qs, pixel_values

    def __len__(self):
        return len(self.questions)

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


question_file = ""
answers_file  = ""
image_folder = ""
questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
# questions = get_chunk(questions, num_chunks, chunk_idx)
answers_file = os.path.expanduser(answers_file)
os.makedirs(os.path.dirname(answers_file), exist_ok=True)
ans_file = open(answers_file, "w")

dataset = CustomDataset(questions, image_folder)

for data in dataset:
    qs, pixel_values = data
    print(qs)
    print(pixel_values.shape)
    break

