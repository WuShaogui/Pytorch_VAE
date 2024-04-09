import torch
import os
from nets.vanilla_vae import VanillaVAE
from PIL import Image
import numpy as np

import cv2
from utils.utils import preprocess_input


def main():
    # 加载模型
    in_channels = 3
    latent_dim = 256
    device = torch.device("cpu")
    net = VanillaVAE(device, in_channels, latent_dim)
    state_dict = torch.load("runs/best.pth")
    net.load_state_dict(state_dict, strict=True)
    net = net.eval()

    # 重建样本阶段
    data_path = "data/mnist_m_train"
    images_path = [
        image_path.replace("\n", "")
        for image_path in open(
            os.path.join(data_path, "val.txt"), encoding="utf-8", mode="r"
        ).readlines()
    ]


    input_shape = (32, 32)
    save_path = "imgs"
    reconstructed_path = os.path.join(save_path, "Reconstructed")
    sample_path = os.path.join(save_path, "Sample")
    if not os.path.exists(reconstructed_path):
        os.mkdir(reconstructed_path)
    if not os.path.exists(sample_path):
        os.mkdir(sample_path)

    for image_path in images_path[:10]:
        img = Image.open(image_path)
        img = img.convert("RGB")
        img = img.resize(input_shape)
        ori_img = img.copy()
        img = np.transpose(preprocess_input(np.array(img, np.float32)), [2, 0, 1])

        input = torch.from_numpy(np.array([img])).to(device)
        recons = net.generate(input)
        gen_sample = (
            np.transpose(recons[0].detach().numpy(), [1, 2, 0]) + 1
        ) / 2  # 反标准化，拿到[0,1)之间的值

        result = np.hstack([np.array(ori_img), gen_sample * 255])
        result = result.astype(np.uint8)
        cv2.imwrite(
            os.path.join(save_path, "Reconstructed", os.path.basename(image_path)),
            result,
        )

    # 随机生成样本
    samples = net.sample(16, device)
    print(torch.min(samples), torch.max(samples))
    for i, sample in enumerate(samples):
        gen_sample = (
            np.transpose(sample.detach().numpy(), [1, 2, 0]) + 1
        ) / 2  # 反标准化，拿到[0,1)之间的值
        gen_sample = gen_sample * 255
        cv2.imwrite(
            os.path.join(save_path, f"Sample/{i}.png"),
            gen_sample,
        )


if __name__ == "__main__":
    main()
