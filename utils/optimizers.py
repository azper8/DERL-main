import numpy as np


# 参考:
# https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/optimizers.py

class Optimizer(object):
    """优化器基类，定义了优化器的基本接口和功能"""

    def __init__(self, theta, epsilon=1e-08):
        """
        初始化优化器
        :param theta: 待优化的参数向量
        :param epsilon: 极小值，用于数值稳定性，防止除以零
        """
        self.theta = theta  # 待优化的参数
        self.epsilon = epsilon  # 数值稳定性的极小值
        self.dim = len(theta)  # 参数维度
        self.t = 0  # 时间步/迭代次数计数器

    def update(self, globalg):
        """
        更新参数
        :param globalg: 全局梯度
        :return: 更新后的参数
        """
        self.t += 1  # 增加迭代计数器
        step = self._compute_step(globalg)  # 计算更新步长
        self.theta += step  # 更新参数
        return self.theta

    def _compute_step(self, globalg):
        """计算参数更新步长(由子类实现)"""
        raise NotImplementedError


class Adam(Optimizer):
    """Adam优化器实现，自适应矩估计优化算法"""

    def __init__(self, theta, stepsize, beta1=0.99, beta2=0.999):
        """
        初始化Adam优化器
        :param theta: 待优化的参数向量
        :param stepsize: 学习率/步长
        :param beta1: 一阶矩估计的指数衰减率
        :param beta2: 二阶矩估计的指数衰减率
        """
        Optimizer.__init__(self, theta)
        self.stepsize = stepsize  # 学习率
        self.beta1 = beta1  # 一阶矩衰减率
        self.beta2 = beta2  # 二阶矩衰减率
        self.m = np.zeros(self.dim, dtype=np.float32)  # 一阶矩估计(动量)
        self.v = np.zeros(self.dim, dtype=np.float32)  # 二阶矩估计(自适应学习率)

    def _compute_step(self, globalg):
        """
        计算Adam更新步长
        :param globalg: 全局梯度
        :return: 参数更新步长
        """
        # 计算偏差校正后的学习率
        a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        # 更新一阶矩估计(动量)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        # 更新二阶矩估计(自适应学习率)
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        # 计算参数更新步长
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step

    def step2(self, gradients):
        """
        另一种更新方式(与update类似但方向相反)
        :param gradients: 梯度
        """
        step = self._compute_step(gradients)  # 计算步长
        self.theta -= step  # 更新参数(注意这里是减号)
        self.t += 1  # 增加迭代计数器