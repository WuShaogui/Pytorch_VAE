import os
import random
import torch
import cv2
import torch.nn as nn
from PIL import Image
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms
from utils.utils import preprocess_input


class VAEDataset(nn.Module):
    def __init__(
        self,
        label_lines: list,
        input_shape: list,
        is_aug=False,
    ):
        super(VAEDataset, self).__init__()

        self.label_lines = label_lines
        self.input_shape = input_shape
        self.is_aug = is_aug

    def rand(self, a=0, b=1):
        return random.random() * (b - a) + a

    def __getitem__(self, index):
        image_path = self.label_lines[index].strip()
        img = Image.open(image_path)

        img = img.convert("RGB")
        img = img.resize(self.input_shape)
        img = np.transpose(preprocess_input(np.array(img, np.float32)), [2, 0, 1])

        return img

    def __len__(self):
        return len(self.label_lines)


if __name__ == "__main__":
    data_path = "xxxx" # 数据集所在文件夹路径，也就是由build_trainning_dtat.py的保存路径
    label_lines = [
        image_path.replace("\n", "")
        for image_path in open(
            os.path.join(data_path, "train.txt"), encoding="utf-8", mode="r"
        ).readlines()
    ]

    input_shape = (512, 512)
    dataset = VAEDataset(label_lines, input_shape)

    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True, pin_memory=True)

    for i, batch in enumerate(train_dataloader):
        image = batch[0]
        image = image.detach().numpy()
        image = image.transpose(1, 2, 0)
        cv2.imwrite("mask.png", image * 255)
        print(image.shape)
        break
