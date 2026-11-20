import torch
import numpy as np


def get_flatten_params(model):
    """
    将模型所有参数展平为单个向量，并返回重建信息
    :param model: PyTorch神经网络实例
    :return: 字典包含:
             "params": 展平后的参数向量
             "lengths": 每个参数在展平向量中的起始和结束位置列表
                        (结束位置不包含)
    """
    param_list = model.get_param_list()  # 获取参数列表
    l = [np.ravel(p) for p in param_list]  # 展平每个参数
    lengths = []  # 记录每个参数在展平向量中的位置
    s = 0  # 起始位置
    for p in l:
        size = p.shape[0]  # 当前参数大小
        lengths.append((s, s + size))  # 记录位置范围
        s += size  # 更新起始位置
    flat = np.concatenate(l)  # 拼接所有展平参数
    return {"params": flat, "lengths": lengths}


def set_flatten_params(flat_params, lengths, model):
    """
    从展平向量恢复模型参数
    :param flat_params: 展平的参数向量
    :param lengths: 每个参数的位置范围列表
    :param model: 要设置参数的模型
    """
    param_lst = []
    # 从展平向量中提取每个参数
    flat_params_lst = [flat_params[s:e] for (s, e) in lengths]
    # 将每个参数重塑为原始形状
    for param, flat_param in zip(model.get_param_list(), flat_params_lst):
        param_lst.append(np.copy(flat_param.reshape(param.shape)))
    # 设置模型参数
    set_param_list(model, param_lst)


def get_param_list(model):
    """
    获取模型参数列表
    :param model: PyTorch神经网络实例
    :return: 模型参数列表(每个参数为numpy数组)
    """
    param_lst = []
    for param in model.parameters():  # 遍历所有参数
        param_lst.append(param.data.numpy())  # 转换为numpy数组
    return param_lst


def set_param_list(model, param_lst: list):
    """
    从列表设置模型参数
    :param model: PyTorch神经网络实例
    :param param_lst: 参数列表(每个参数为numpy数组)
    """
    lst_idx = 0
    for param in model.parameters():  # 遍历所有参数
        # 将numpy数组转换为tensor并设置参数
        param.data = torch.tensor(param_lst[lst_idx]).float()
        lst_idx += 1


def compute_ranks(x):
    """
    计算数组中元素的排名(从0开始)
    注意: 与scipy.stats.rankdata不同，后者返回[1, len(x)]的排名
    :param x: 输入数组(一维)
    :return: 排名数组(与x同形状)
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)  # 初始化排名数组
    ranks[x.argsort()] = np.arange(len(x))  # 计算排名
    return ranks


def compute_centered_ranks(x):
    """
    计算中心化排名(归一化到[-0.5, 0.5])
    :param x: 输入数组
    :return: 中心化后的排名
    """
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)  # 归一化到[0,1]
    y -= .5  # 中心化到[-0.5,0.5]
    return y


@torch.no_grad()
def xavier_init(m):
    """
    Xavier初始化(应用于线性层)
    :param m: 模型或层
    """
    if isinstance(m, torch.nn.Linear):  # 如果是线性层
        torch.nn.init.xavier_normal_(m.weight)  # Xavier初始化权重
        m.bias.fill_(0.0)  # 偏置初始化为0