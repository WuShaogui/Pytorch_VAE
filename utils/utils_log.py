import datetime
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image

plt.switch_backend("agg")
from torch.utils.tensorboard import SummaryWriter

from pylab import mpl

# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
colors = ["b", "g", "r", "c", "m", "y", "k"]


class TrainingHistory(object):
    def __init__(self, log_dir, net, input_shape):
        super(TrainingHistory, self).__init__()

        self.log_dir = log_dir

        self.train_loss = {}
        self.val_loss = {}

        # tensorboard显示布局
        layout = {
            "ABCDE": {"loss": ["Multiline", ["loss/train", "loss/validation"]]},
        }
        self.writer = SummaryWriter(self.log_dir)
        self.writer.add_custom_scalars(layout)

        try:
            if len(input_shape) == 2:
                dummy_input = torch.randn((1, 3, input_shape[0], input_shape[1])).cuda()
            elif len(input_shape) == 3:
                dummy_input = torch.randn(
                    (1, input_shape[0], input_shape[1], input_shape[2])
                ).cuda()
            self.writer.add_graph(net, dummy_input)
        except:
            pass

    def append_loss(self, cur_epoch: int, loss: float, is_training=True, prefix=""):
        if is_training:
            if prefix in self.train_loss.keys():
                self.train_loss[prefix].append(loss)
            else:
                self.train_loss[prefix] = [loss]

            with open(os.path.join(self.log_dir, "train_loss.txt"), "a") as fw:
                fw.write(str(self.train_loss[prefix]) + "\n")
            self.writer.add_scalar("loss/train", loss, cur_epoch)
        else:
            if prefix in self.val_loss.keys():
                self.val_loss[prefix].append(loss)
            else:
                self.val_loss[prefix] = [loss]
            with open(os.path.join(self.log_dir, "val_loss.txt"), "a") as fw:
                fw.write(str(self.val_loss) + "\n")
            self.writer.add_scalar("loss/validation", loss, cur_epoch)

    def append_lr(self, cur_epoch: int, lr: float):
        self.writer.add_scalar("learnning rate", lr, cur_epoch)

    def plot_learning_curves(self, cur_epoch):
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("epochs")
        ax1.set_ylabel("loss")

        lns = []
        for i, prefix in enumerate(self.train_loss.keys()):
            ln1 = ax1.plot(
                np.arange(len(self.train_loss[prefix])),
                self.train_loss[prefix],
                color=colors[i % len(colors)],
            )
            lns += ln1
        for i, prefix in enumerate(self.val_loss.keys()):
            ln2 = ax1.plot(
                np.arange(len(self.val_loss[prefix])),
                self.val_loss[prefix],
                color=colors[i % len(colors)],
                linestyle="dashed",
            )
            lns += ln2

        plt.legend(lns, list(self.train_loss.keys()) + list(self.val_loss.keys()))
        plt.tight_layout()
        plt.savefig("{}/learning_curve.png".format(self.log_dir), bbox_inches="tight")
        plt.close("all")

        self.writer.add_figure("learning curve", fig, cur_epoch)

    def close(self):
        self.writer.close()
