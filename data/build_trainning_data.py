"""
语义分割文件布局

原始文件布局：
布局1：
- xxxx
    - xxx.png
    - xxx.json
    - yyy.bmp
    - yyy.json
    ...


目标文件夹：
- ooo
    - train 
    - val
    - train.txt
    - val.txt
    - info.txt
"""

# 计划
"""
1. 图片-json对的扫描
2. 解析json的特定label为mask
3. 将图片拷贝到images文件夹，mask生成到masks文件夹
4. 生成train.txt、val.txt、info.txt
"""

import os
import os.path as osp
from glob import glob
import pathlib
import random
import json
import time
import numpy as np
import cv2
import shutil
from PIL import Image
import tqdm


def scan_images(images_dir):
    """扫描文件夹，返回图片绝对路径

    Args:
        images_dir (str): 文件夹路径

    Returns:
        list: 所有图片的绝对路径
    """
    images_path = []
    for ext in ["png", "jpg", "jpeg", "bmp"]:
        ext_images_path = pathlib.Path(images_dir).glob("**/*." + ext)
        for image_path in ext_images_path:
            images_path.append(str(image_path))

    return images_path


def main(data_dir, save_dir, train_roate=0.9, seed=0):
    random.seed(seed)

    # 扫描文件夹，得到训练集、验证集
    print("scan images...")
    train_file, val_file = [], []
    images_path = scan_images(data_dir)
    random.shuffle(images_path)
    train_index = int(train_roate * len(images_path))
    train_file = images_path[:train_index]
    val_file = images_path[train_index:]

    # 拷贝文件
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    if not osp.exists(osp.join(save_dir, "train")):
        os.makedirs(osp.join(save_dir, "train"))
    if not osp.exists(osp.join(save_dir, "val")):
        os.makedirs(osp.join(save_dir, "val"))

    with open(osp.join(save_dir, "train.txt"), encoding="utf-8", mode="w") as fw:
        for class_idx in train_file:
            save_image_path = osp.join(save_dir, "train\\", osp.basename(class_idx))
            shutil.copyfile(class_idx, save_image_path)
            fw.write("{}\n".format(save_image_path))

    with open(osp.join(save_dir, "val.txt"), encoding="utf-8", mode="w") as fw:
        for class_idx in val_file:
            save_image_path = osp.join(save_dir, "val\\", osp.basename(class_idx))
            shutil.copyfile(class_idx, save_image_path)
            fw.write("{}\n".format(save_image_path))

    print("train num:{}  val num:{}".format(len(train_file), len(val_file)))
    print("build data done")


if __name__ == "__main__":
    data_dir = r"C:\Users\Administrator\Downloads\中航阳极分类数据"
    save_dir = r"C:\Users\Administrator\Documents\MyCodes\Pytorch_VAE\imgs"
    main(data_dir, save_dir)
