import numpy as np
from typing import Tuple, Union


class RunningMeanStd(object):
    """实时计算数据流的均值和标准差"""

    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """
        初始化运行统计量计算器
        :param epsilon: 极小值，用于数值稳定性
        :param shape: 数据流的形状
        参考: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        """
        self.mean = np.zeros(shape, np.float64)  # 均值
        self.var = np.ones(shape, np.float64)  # 方差
        self.count = epsilon  # 样本计数(初始化为极小值防止除零)

    def copy(self) -> "RunningMeanStd":
        """
        创建当前对象的深拷贝
        :return: RunningMeanStd的新实例
        """
        new_object = RunningMeanStd(shape=self.mean.shape)
        new_object.mean = self.mean.copy()  # 复制均值
        new_object.var = self.var.copy()  # 复制方差
        new_object.count = float(self.count)  # 复制计数
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        """
        合并另一个RunningMeanStd对象的统计量
        :param other: 要合并的另一个RunningMeanStd对象
        """
        self.update_from_moments(other.mean, other.var, other.count)

    def update(self, arr: np.ndarray) -> None:
        """
        用新数据更新统计量
        :param arr: 新数据数组
        """
        batch_mean = np.mean(arr, axis=0)  # 计算批次均值
        batch_var = np.var(arr, axis=0)  # 计算批次方差
        batch_count = arr.shape[0]  # 获取批次大小
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: Union[int, float]) -> None:
        """
        用矩(moments)更新统计量
        :param batch_mean: 批次均值
        :param batch_var: 批次方差
        :param batch_count: 批次大小
        """
        delta = batch_mean - self.mean  # 计算均值差
        tot_count = self.count + batch_count  # 总样本数

        # 计算新的均值(加权平均)
        new_mean = self.mean + delta * batch_count / tot_count

        # 计算合并方差
        m_a = self.var * self.count  # 当前方差×当前计数
        m_b = batch_var * batch_count  # 批次方差×批次计数
        # 合并二阶矩
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)  # 新方差

        # 更新总计数
        new_count = batch_count + self.count

        # 更新内部状态
        self.mean = new_mean
        self.var = new_var
        self.count = new_count