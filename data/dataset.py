import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from utils.utils import preprocess_input


class VAEDataset(nn.Module):
    def __init__(self, label_lines: list, input_shape: tuple, is_aug=False):
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
    label_lines = [
        image_path.replace("\n", "")
        for image_path in open(
            r"imgs\train.txt", encoding="utf-8", mode="r"
        ).readlines()
    ]
    input_shape = (64, 64)
    classes_idx = [0, 1]
    is_aug = False
    dataset = VAEDataset(label_lines, input_shape, is_aug)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=True, num_workers=4
    )

    for i, batch in enumerate(dataloader):
        image = batch[0]
        image = image.detach().numpy()
        image = image.transpose(1, 2, 0)
        cv2.imwrite("mask.png", image * 255)
        print(image.shape)
        break
