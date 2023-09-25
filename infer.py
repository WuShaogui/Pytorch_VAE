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
        "/home/wushaogui/MyCodes/Pytorch_VAE/imgs/中航阳极分类数据/val/内侧-2022.10.14 11.04.15.483-破损面积287.222-2帧：2219-序号：28-高21.52-宽32.55-.bmp",
        "/home/wushaogui/MyCodes/Pytorch_VAE/imgs/中航阳极分类数据/val/内侧-2022.10.14 02.18.49.828-破损面积41.518-2帧：2277-序号：1-高21.84-宽44.15-.bmp",
        "/home/wushaogui/MyCodes/Pytorch_VAE/imgs/中航阳极分类数据/val/内侧-2022.10.14 19.59.50.062-破损面积246.851-2帧：1368-序号：31-高19.41-宽43.58-.bmp",
        "/home/wushaogui/MyCodes/Pytorch_VAE/imgs/中航阳极分类数据/val/内侧-EA数_24-2022.10.14 03.31.41.743-气泡 -面积1.930-分切内侧-帧710-宽1.360-高1.650-.bmp",
        "/home/wushaogui/MyCodes/Pytorch_VAE/imgs/中航阳极分类数据/val/内侧-EA数_48-2022.10.16 04.13.10.432-AT9漏金属-面积0.387-宽0.275-分切内侧-帧1367-.jpeg",
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
