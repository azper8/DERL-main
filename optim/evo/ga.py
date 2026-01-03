import torch
import random
import numpy as np
from copy import deepcopy
from optim.base_optim import BaseOptim
from utils.policy_dict import agent_policy
from utils.torch_util import get_flatten_params, set_flatten_params


class GeneticOptimizer(BaseOptim):
    """遗传算法优化器（用于神经网络权重优化）"""

    def __init__(self, config):
        """
        参数:
            config (dict): 包含:
                - population_size: 种群大小
                - crossover_rate: 交叉概率(默认0.7)
                - mutation_rate: 变异概率(默认0.1)
                - mutation_scale: 变异幅度(默认0.01)
                - elite_frac: 精英保留比例(默认0.2)
        """
        super().__init__()
        self.population_size = config["population_size"]
        self.crossover_rate = config.get("crossover_rate", 0.7)
        self.mutation_rate = config.get("mutation_rate", 0.1)
        self.mutation_scale = config.get("mutation_scale", 0.01)
        self.elite_frac = config.get("elite_frac", 0.2)

        self.population = []
        self.agent_ids = None
        self.mu_model = None
        self.best_fitness = -float('inf')

    def init_population(self, policy: torch.nn.Module, env):
        """初始化种群"""
        self.agent_ids = env.get_agent_ids()
        policy.norm_init()
        self.mu_model = policy

        # 生成初始种群
        base_params = get_flatten_params(self.mu_model)['params']
        self.population = [
            base_params + np.random.normal(0, 0.1, base_params.shape)
            for _ in range(self.population_size)
        ]
        return self._create_policy_objects()

    # def init_population(self, policy: torch.nn.Module, env):
    #     """初始化种群"""
    #     self.agent_ids = env.get_agent_ids()
    #     self.mu_model = deepcopy(policy)
    #     self.mu_model.norm_init()
    #
    #     # 生成初始种群（每个个体都是独立初始化的策略）
    #     self.population = []
    #     for _ in range(self.population_size):
    #         # 创建新策略并独立初始化
    #         new_policy = deepcopy(policy)
    #         new_policy.norm_init()
    #         # 获取参数并存储
    #         params = get_flatten_params(new_policy)['params']
    #         self.population.append(params)
    #
    #     return self._create_policy_objects()

    def _create_policy_objects(self):
        """将参数列表转换为策略对象列表"""
        perturbations = []
        for i, params in enumerate([get_flatten_params(self.mu_model)['params']] + self.population):
            policy = deepcopy(self.mu_model)
            set_flatten_params(params, get_flatten_params(policy)['lengths'], policy)
            policy.set_policy_id(i)
            perturbations.append(agent_policy(self.agent_ids, policy))
        return perturbations

    def crossover(self, parent1, parent2):
        """单点交叉"""
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, len(parent1) - 1)
            child = np.concatenate([
                parent1[:crossover_point],
                parent2[crossover_point:]
            ])
            return child
        return parent1

    def mutate(self, individual):
        """高斯变异"""
        mask = np.random.random(size=individual.shape) < self.mutation_rate
        noise = np.random.normal(0, self.mutation_scale, size=individual.shape)
        return individual + mask * noise

    def select(self, fitness_scores):
        """精英选择+轮盘赌选择"""
        elite_size = int(self.population_size * self.elite_frac)
        ranked = sorted(zip(self.population, fitness_scores), key=lambda x: -x[1])

        # 精英保留
        elites = [x[0] for x in ranked[:elite_size]]

        # 轮盘赌选择
        probs = np.array(fitness_scores) - min(fitness_scores)
        probs = probs / (probs.sum() + 1e-8)
        selected = np.random.choice(
            range(len(self.population)),
            size=self.population_size - elite_size,
            p=probs
        )
        return elites + [self.population[i] for i in selected]

    def next_population(self, assemble, results, g):
        """生成新一代种群"""
        rewards = np.array(results['rewards'].tolist())
        fitness = rewards[1:]  # 忽略第一个原始模型

        # 更新最佳模型
        if max(fitness) > self.best_fitness:
            self.best_fitness = max(fitness)
            best_idx = np.argmax(fitness) + 1
            set_flatten_params(
                self.population[best_idx - 1],
                get_flatten_params(self.mu_model)['lengths'],
                self.mu_model
            )

        # 1. 选择
        selected = self.select(fitness)

        # 2. 交叉和变异
        new_population = []
        while len(new_population) < self.population_size:
            # 锦标赛选择
            p1, p2 = random.sample(selected, 2)
            child = self.crossover(p1, p2)
            child = self.mutate(child)
            new_population.append(child)

        self.population = new_population

        # 3. 自适应调整变异率
        if g % 10 == 0:
            diversity = np.mean([np.std(ind) for ind in zip(*self.population)])
            self.mutation_rate = max(0.01, min(0.5, 0.1 * (1 + np.tanh(diversity - 1))))

        return self._create_policy_objects(), self.mutation_rate, max(fitness)

    def get_elite_model(self):
        """获取最佳模型"""
        return self.mu_model


def compute_ranks(x):
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y