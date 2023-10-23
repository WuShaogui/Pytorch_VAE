import datetime
import os
import sys

import numpy as np
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from data.dataset import VAEDataset
from nets.vanilla_vae import VanillaVAE, weights_init
from utils.utils import set_seed
from utils.utils import get_lr
from utils.utils_log import TrainingHistory

import os.path as osp


if sys.gettrace() is not None:
    # Code 1: This code will be executed when debugging
    print("Debugging")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    # Code 2: This code will be executed when running without debugging
    print("Running")
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if __name__ == "__main__":

    set_seed(seed=0)

    # TODO 加载数据集
    data_path = "/home/wushaogui/MyCodes/Pytorch_VAE/imgs/反光"
    train_label_lines = [
        image_path.replace("\n", "")
        for image_path in open(
            os.path.join(data_path, "train.txt"), encoding="utf-8", mode="r"
        ).readlines()
    ]
    val_label_lines = [
        image_path.replace("\n", "")
        for image_path in open(
            os.path.join(data_path, "val.txt"), encoding="utf-8", mode="r"
        ).readlines()
    ]
    input_shape = (128, 128)
    batch_size = 16
    train_dataset = VAEDataset(train_label_lines, input_shape)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_dataset = VAEDataset(train_label_lines, input_shape)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    # TODO 加载模型
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    in_channels = 3
    latent_dim = 256
    net = VanillaVAE(device, in_channels, latent_dim)
    weights_init(net)
    net = net.to(device)

    # 定义优化函数
    opt = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.98)

    # 训练模型
    log_subdir = datetime.datetime.strftime(
        datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S"
    )
    log_dir = os.path.abspath(os.path.join("runs", log_subdir))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_subdir = datetime.datetime.strftime(
        datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S"
    )
    log_dir = osp.abspath(osp.join("runs", log_subdir))
    # 初始化日志目录
    record = TrainingHistory(log_dir, net, input_shape)

    min_loss = np.inf
    num_epoches = 300
    for epoch in range(num_epoches):
        train_total_loss = 0
        total_reconstruction_loss = 0
        total_kld_loss = 0

        net = net.train()
        for i, batch in enumerate(train_dataloader):
            img = batch
            img = img.to(device)
            # 前向推理
            reconstructed_x, input, mu, log_var = net(img)
            loss = net.loss_function(reconstructed_x, input, mu, log_var, M_N=0.002)
            train_loss = loss["loss"]

            # 后向更新
            opt.zero_grad()
            train_loss.backward()
            opt.step()

            # 统计损失
            reconstruction_loss = loss["Reconstruction_Loss"]
            kld_loss = loss["KLD"]
            total_reconstruction_loss += reconstruction_loss.item()
            total_kld_loss += kld_loss.item()
            train_total_loss += train_loss.item()

            if i % 50 == 0:
                print(
                    "epoch:{} train loss:{:.4f} reconstruction_loss:{:.4f} kld_loss:{:.4f} lr:{}".format(
                        epoch,
                        train_total_loss / (i + 1),
                        total_reconstruction_loss / (i + 1),
                        total_kld_loss / (i + 1),
                        get_lr(opt),
                    )
                )
        record.append_loss(epoch, train_total_loss / (i + 1), prefix="train_loss")
        record.append_loss(
            epoch,
            total_reconstruction_loss / (i + 1),
            prefix="train_reconstruction_loss",
        )
        record.append_loss(epoch, total_kld_loss / (i + 1), prefix="train_kld_loss")

        # 评估阶段
        net = net.eval()
        val_total_loss = 0
        val_reconstruction_loss = 0
        val_kld_loss = 0
        for i, batch in enumerate(val_dataloader):
            img = batch
            img = img.to(device)
            results = net(img)
            val_loss = net.loss_function(
                *results,
                M_N=1.0,
            )
            val_total_loss += val_loss["loss"].item()
            reconstruction_loss = val_loss["Reconstruction_Loss"]
            val_reconstruction_loss += reconstruction_loss.item()
            kld_loss = val_loss["KLD"]
            val_kld_loss += kld_loss.item()

            if i % 50 == 0:
                print(
                    "epoch:{} val loss:{:.4f} reconstruction_loss:{:.4f} kld_loss:{:.4f} lr:{}".format(
                        epoch,
                        val_total_loss / (i + 1),
                        reconstruction_loss.item(),
                        kld_loss.item(),
                        get_lr(opt),
                    )
                )
        record.append_loss(
            epoch, val_total_loss / (i + 1), is_training=False, prefix="val_loss"
        )
        record.append_loss(
            epoch,
            val_reconstruction_loss / (i + 1),
            is_training=False,
            prefix="val_reconstruction_loss",
        )
        record.append_loss(
            epoch,
            val_kld_loss / (i + 1),
            is_training=False,
            prefix="val_kld_loss",
        )

        if val_total_loss / (i + 1) < min_loss:
            torch.save(net.eval().state_dict(), os.path.join("runs/best.pth"))

        # # 重建样本阶段
        # if not os.path.exists(os.path.join(log_dir, "Reconstructed")):
        #     os.makedirs(os.path.join(log_dir, "Reconstructed"))
        # test_input = next(iter(val_dataloader))
        # test_input = test_input.to(device)
        # recons = net.generate(test_input)
        # vutils.save_image(
        #     recons.data,
        #     "{}/Reconstructed/epoch-{}.png".format(log_dir, epoch),
        #     normalize=True,
        #     nrow=4,
        # )

        # # 随机生成样本阶段
        # if not os.path.exists(os.path.join(log_dir, "Samples")):
        #     os.makedirs(os.path.join(log_dir, "Samples"))
        # samples = net.sample(16, device)
        # vutils.save_image(
        #     samples.cpu().data,
        #     "{}/Samples/epoch-{}.png".format(log_dir, epoch),
        #     normalize=True,
        #     nrow=4,
        # )

        # 学习率衰减
        lr_scheduler.step()
        record.plot_learning_curves(epoch)
