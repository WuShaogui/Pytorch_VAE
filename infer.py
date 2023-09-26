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
    images_path = {
        "xxx", #待重建的文件路径
    }
    input_shape = (128, 128)
    save_path = "imgs"

    for image_path in images_path:
        img = Image.open(image_path)
        img = img.convert("RGB")
        img = img.resize(input_shape)
        ori_img = img.copy()
        img = np.transpose(preprocess_input(np.array(img, np.float32)), [2, 0, 1])

        input = torch.from_numpy(np.array([img])).to(device)
        recons = net.generate(input)

        result = np.hstack(
            [
                np.array(ori_img),
                np.transpose(recons[0].detach().numpy(), [1, 2, 0]) * 255,
            ]
        )
        result = result.astype(np.uint8)
        cv2.imwrite(
            os.path.join(save_path, "Reconstructed", os.path.basename(image_path)),
            result,
        )

    # 随机生成样本
    samples = net.sample(16, device)
    for i, sample in enumerate(samples):
        cv2.imwrite(
            os.path.join(save_path, f"Sample/{i}.png"),
            np.transpose(sample.detach().numpy(), [1, 2, 0]) * 255,
        )


if __name__ == "__main__":
    main()
