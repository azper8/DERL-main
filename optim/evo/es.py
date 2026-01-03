"""
ES (Evolution Strategies) 算法实现 - OpenAI风格
用于更新策略参数的进化策略优化器
"""
import torch
import numpy as np
from copy import deepcopy
from optim.base_optim import BaseOptim
from utils.policy_dict import agent_policy
from utils.torch_util import get_flatten_params, set_flatten_params


class ESOpenAI(BaseOptim):
    """OpenAI风格的进化策略优化器"""

    def __init__(self, config):
        """
        初始化ES优化器

        参数:
            config (dict): 配置字典，包含:
                - name: 优化器名称
                - sigma_init: 初始噪声标准差
                - sigma_decay: 噪声衰减系数
                - learning_rate: 学习率
                - reinforce_learning_rate: 强化学习率
                - population_size: 种群大小
                - reward_shaping: 是否使用奖励塑形
                - reward_norm: 是否归一化奖励
        """
        super(ESOpenAI, self).__init__()
        self.name = config["name"]  # 优化器名称
        self.sigma_init = config["sigma_init"]  # 初始噪声标准差
        self.sigma_curr = self.sigma_init  # 当前噪声标准差
        self.sigma_decay = config["sigma_decay"]  # 噪声衰减系数
        self.learning_rate = config["learning_rate"]  # 基础学习率
        self.reinforce_learning_rate = config["reinforce_learning_rate"]  # 强化学习率
        self.population_size = config["population_size"]  # 种群大小
        self.reward_shaping = config['reward_shaping']  # 是否使用奖励塑形
        self.reward_norm = config['reward_norm']  # 是否归一化奖励

        self.epsilons = []  # 保存每个模型的epsilon噪声向量
        self.agent_ids = None  # 代理ID列表
        self.mu_model = None  # 当前均值模型(主策略)
        self.mu_model_flatten_params = None  # 扁平化的模型参数
        self.optimizer = None  # 优化器(Adam)
        self.population = None

    def init_population(self, policy: torch.nn.Module, env):
        """
        初始化种群: θ_t 和 (θ_t + σϵ_i)

        参数:
            policy: 初始策略模型
            env: 环境对象

        返回:
            list: 包含扰动策略的列表
        """
        # 1. 初始化主策略θ_t
        self.agent_ids = env.get_agent_ids()  # 获取代理ID
        # policy.norm_init()  # 标准化初始化
        self.mu_model = policy  # 保存主策略
        # 获取扁平化参数并设置为可训练
        self.mu_model_flatten_params = torch.tensor(
            get_flatten_params(self.mu_model)['params'],
            requires_grad=True,
            dtype=torch.float64
        )
        # print(len(self.mu_model_flatten_params))
        # 初始化优化器
        self.optimizer = torch.optim.Adam(
            [self.mu_model_flatten_params],
            lr=self.learning_rate
        )

        # 2. 初始化扰动策略(θ_t + σϵ_i)
        self.population = self.init_perturbations(
            self.agent_ids,
            self.mu_model,
            self.sigma_curr,
            self.population_size
        )
        return self.population

    def init_perturbations(self, agent_ids, mu_model, sigma, pop_size):
        """
        生成扰动策略种群（使用镜像采样）

        参数:
            agent_ids: 代理ID列表
            mu_model: 主策略模型
            sigma: 当前噪声标准差
            pop_size: 种群大小（实际生成pop_size+1个，包含主策略）

        返回:
            list: 扰动策略列表
        """
        perturbations = []  # 扰动策略列表
        self.epsilons = []  # 重置噪声向量列表

        # 添加主策略到种群中(用于评估)
        perturbations.append(agent_policy(agent_ids, mu_model))

        # 为主策略添加零噪声(实现技巧)
        zero_eps = deepcopy(mu_model)
        zero_eps.zero_init()  # 零初始化
        zero_eps_param_lst = get_flatten_params(zero_eps)
        self.epsilons.append(zero_eps_param_lst['params'])

        # 使用镜像采样：生成n/2个噪声向量，然后创建正负对
        half_pop = pop_size // 2
        for i in range(half_pop):
            # 生成一个噪声向量
            perturbed_policy_plus = deepcopy(mu_model)  # 复制主策略
            perturbed_policy_minus = deepcopy(mu_model)  # 复制主策略
            
            # 设置策略ID
            perturbed_policy_plus.set_policy_id(2 * i + 1)  # 奇数ID
            perturbed_policy_minus.set_policy_id(2 * i + 2)  # 偶数ID

            # 获取参数
            perturbed_policy_param_lst_plus = get_flatten_params(perturbed_policy_plus)
            perturbed_policy_param_lst_minus = get_flatten_params(perturbed_policy_minus)
            
            # 生成噪声向量
            epsilon = np.random.normal(size=perturbed_policy_param_lst_plus['params'].shape)
            
            # 创建正负扰动：θ_t + σϵ 和 θ_t - σϵ
            perturbed_policy_param_updated_plus = (
                perturbed_policy_param_lst_plus['params'] + epsilon * sigma
            )
            perturbed_policy_param_updated_minus = (
                perturbed_policy_param_lst_minus['params'] - epsilon * sigma
            )

            # 更新扰动策略参数
            set_flatten_params(
                perturbed_policy_param_updated_plus,
                perturbed_policy_param_lst_plus['lengths'],
                perturbed_policy_plus
            )
            set_flatten_params(
                perturbed_policy_param_updated_minus,
                perturbed_policy_param_lst_minus['lengths'],
                perturbed_policy_minus
            )

            # 添加到种群
            perturbations.append(agent_policy(agent_ids, perturbed_policy_plus))
            perturbations.append(agent_policy(agent_ids, perturbed_policy_minus))
            
            # 保存噪声向量（正负对称）
            self.epsilons.append(epsilon)   # 对应正扰动
            self.epsilons.append(-epsilon)  # 对应负扰动

        return perturbations

    def next_population(self, results, g):
        """
        生成下一代种群

        参数:
            results: 评估结果DataFrame
            g: 当前代数

        返回:
            tuple: (新种群, 当前sigma值, 最佳奖励)
        """
        rewards = results['rewards'].tolist()
        makespans = results['makespan'].tolist()
        overrates = results['violation_rate'].tolist()
        best_reward_idx = np.argmax(rewards)
        best_reward = rewards[best_reward_idx]
        best_makespan = makespans[best_reward_idx]
        best_overrate = overrates[best_reward_idx]

        # 获取当前奖励最大的个体
        best_individual = self.population[best_reward_idx]

        rewards = np.array(rewards)

        # 奖励塑形
        if self.reward_shaping:
            rewards = compute_centered_ranks(rewards)

        # 奖励归一化
        if self.reward_norm:
            r_std = rewards.std()
            rewards = (rewards - rewards.mean()) / r_std

        # 计算更新因子
        # 注意：由于使用镜像采样，有效种群大小是pop_size+1
        update_factor = 1 / ((len(self.epsilons) - 1) * self.sigma_curr)
        update_factor *= -1.0  # 适配最小化问题

        # 计算梯度更新: sum(F_j * epsilon_j)
        grad_param_list = np.sum(
            np.array(self.epsilons) * rewards.reshape(rewards.shape[0], 1),
            axis=0
        )
        grad_param_list *= update_factor
        # print("Gradient norm:", np.linalg.norm(grad_param_list))
        # 参数更新
        self.mu_model_flatten_params.grad = torch.tensor(
            grad_param_list,
            dtype=torch.float64
        )
        self.optimizer.step()

        # 更新主策略参数
        flatten_params = self.mu_model_flatten_params.clone()
        set_flatten_params(
            flatten_params.detach().numpy(),
            get_flatten_params(self.mu_model)['lengths'],
            self.mu_model
        )

        # 生成新一代扰动策略
        self.population = self.init_perturbations(
            self.agent_ids,
            self.mu_model,
            self.sigma_curr,
            self.population_size
        )

        # 噪声衰减(不低于0.01)
        if self.sigma_curr >= 0.01:
            self.sigma_curr *= self.sigma_decay

        return self.population, self.sigma_curr, best_reward, best_makespan, best_overrate, best_individual

    def get_elite_model(self):
        """获取精英模型(当前主策略)"""
        return self.mu_model


def compute_ranks(x):
    """
    计算排名(范围[0, len(x)))

    注意: 不同于scipy.stats.rankdata(返回[1, len(x)])
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    """计算中心化排名(将奖励转换到[-0.5, 0.5]范围)"""
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y
