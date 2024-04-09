import numpy as np
import random
import torch


def set_seed(seed=0):
    """初始化随机数

    Args:
        seed (int, optional): 随机数种子. Defaults to 0.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def preprocess_input(image):
    image /= 255.0
    image -= np.array([0.5, 0.5, 0.5])
    image /= np.array([0.5, 0.5, 0.5])
    return image


def get_lr(opt):
    """获取优化器的学习率

    Args:
        opt (torch.opti.optimer): 优化器

    Returns:
        float: 学习率值
    """
    for par in opt.param_groups:
        return par["lr"]
