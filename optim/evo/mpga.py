import random
import numpy as np
from copy import deepcopy
from optim.base_optim import BaseOptim
from utils.policy_dict import agent_policy
from utils.torch_util import get_flatten_params, set_flatten_params


'''class GeneticOptimizer(BaseOptim):
    """遗传算法优化器（用于神经网络权重优化）支持多策略"""

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

        self.population = []  # 每个元素现在是(routing_params, sequencing_params)的元组
        self.agent_ids = None
        self.mu_models = None  # 现在是(routing_model, sequencing_model)的元组
        self.best_fitness = -float('inf')

    def init_population(self, policies: tuple, env):
        """初始化种群
        参数:
            policies: (routing_policy, sequencing_policy) 的元组
            env: 环境对象
        """
        self.agent_ids = env.get_agent_ids()
        # 深拷贝并初始化两个模型
        routing_policy, sequencing_policy = policies
        self.mu_models = (
            deepcopy(routing_policy),
            deepcopy(sequencing_policy)
        )
        self.mu_models[0].norm_init()
        self.mu_models[1].norm_init()

        # 生成初始种群（每个个体都是独立初始化的策略对）
        self.population = []
        for _ in range(self.population_size):
            # 创建新策略并独立初始化
            new_routing = deepcopy(routing_policy)
            new_sequencing = deepcopy(sequencing_policy)
            new_routing.norm_init()
            new_sequencing.norm_init()

            # 获取参数并存储为元组
            routing_params = get_flatten_params(new_routing)['params']
            sequencing_params = get_flatten_params(new_sequencing)['params']
            self.population.append((routing_params, sequencing_params))

        return self._create_policy_objects()

    def _create_policy_objects(self):
        """将参数列表转换为策略对象列表"""
        perturbations = []

        # 首先添加原始模型对
        base_routing = deepcopy(self.mu_models[0])
        base_sequencing = deepcopy(self.mu_models[1])
        base_routing.set_policy_id(0)
        base_sequencing.set_policy_id(0)
        perturbations.append(agent_policy(
            self.agent_ids,
            (base_routing, base_sequencing)
        ))

        # 添加种群中的个体
        for i, (routing_params, sequencing_params) in enumerate(self.population):
            # 创建路由策略
            routing_policy = deepcopy(self.mu_models[0])
            set_flatten_params(
                routing_params,
                get_flatten_params(routing_policy)['lengths'],
                routing_policy
            )
            routing_policy.set_policy_id(i + 1)

            # 创建排序策略
            sequencing_policy = deepcopy(self.mu_models[1])
            set_flatten_params(
                sequencing_params,
                get_flatten_params(sequencing_policy)['lengths'],
                sequencing_policy
            )
            sequencing_policy.set_policy_id(i + 1)

            perturbations.append(agent_policy(
                self.agent_ids,
                (routing_policy, sequencing_policy)
            ))

        return perturbations

    def crossover(self, parent1, parent2):
        """交叉操作: 随机选择一个策略进行交叉，另一个策略直接交换"""
        routing1, sequencing1 = parent1
        routing2, sequencing2 = parent2

        if random.random() < self.crossover_rate:
            # 随机选择要交叉的策略
            if random.random() < 0.5:
                return (routing1, sequencing2)
            else:
                return (routing2, sequencing1)
        return parent1

    def mutate(self, individual):
        """对两个策略分别进行高斯变异"""
        routing, sequencing = individual

        # 变异路由策略
        mask = np.random.random(size=routing.shape) < self.mutation_rate
        noise = np.random.normal(0, self.mutation_scale, size=routing.shape)
        mutated_routing = routing + mask * noise

        # 变异排序策略
        mask = np.random.random(size=sequencing.shape) < self.mutation_rate
        noise = np.random.normal(0, self.mutation_scale, size=sequencing.shape)
        mutated_sequencing = sequencing + mask * noise

        return (mutated_routing, mutated_sequencing)

    def select(self, fitness_scores):
        """精英选择+轮盘赌选择"""
        elite_size = int(self.population_size * self.elite_frac)
        ranked = sorted(zip(self.population, fitness_scores), key=lambda x: -x[1])

        # 精英保留
        elites = [x[0] for x in ranked[:elite_size]]

        # 轮盘赌选择
        probs = np.array(fitness_scores) - min(fitness_scores)
        probs = probs / (probs.sum() + 1e-8)
        # 如果 probs 全 0（所有适应度相同），改用均匀分布
        if np.all(probs == 0):
            probs = np.ones_like(probs) / len(probs)
            print("prob全0")
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
        original_fitness = rewards[0]  # 原始模型的适应度

        # 更新最佳模型 - 考虑原始模型和种群中的最佳个体
        current_best_fitness = max(np.max(fitness), original_fitness)

        if current_best_fitness > self.best_fitness:
            self.best_fitness = current_best_fitness

            # 如果原始模型更好，则不需要更新
            if original_fitness == current_best_fitness:
                pass  # 保持self.mu_models不变
            else:
                # 更新为种群中的最佳个体
                best_idx = np.argmax(fitness) + 1
                best_routing, best_sequencing = self.population[best_idx - 1]

                # 更新路由模型
                set_flatten_params(
                    best_routing,
                    get_flatten_params(self.mu_models[0])['lengths'],
                    self.mu_models[0]
                )
                # 更新排序模型
                set_flatten_params(
                    best_sequencing,
                    get_flatten_params(self.mu_models[1])['lengths'],
                    self.mu_models[1]
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

        return self._create_policy_objects(), self.mutation_rate, max(fitness)

    def get_elite_model(self):
        """获取最佳模型对"""
        return self.mu_models'''


class GeneticOptimizer(BaseOptim):
    """遗传算法优化器（用于神经网络权重优化）支持多策略"""

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

        self.population = []  # 每个元素现在是(routing_params, sequencing_params)的元组
        self.agent_ids = None
        self.mu_models = None  # 现在是(routing_model, sequencing_model)的元组
        self.best_fitness = -float('inf')

    def init_population(self, policies: tuple, env):
        """初始化种群
        参数:
            policies: (routing_policy, sequencing_policy) 的元组
            env: 环境对象
        """
        self.agent_ids = env.get_agent_ids()
        # 深拷贝并初始化两个模型
        routing_policy, sequencing_policy = policies
        self.mu_models = (
            deepcopy(routing_policy),
            deepcopy(sequencing_policy)
        )
        self.mu_models[0].norm_init()
        self.mu_models[1].norm_init()

        # 生成初始种群（每个个体都是独立初始化的策略对）
        self.population = []
        for _ in range(self.population_size):
            # 创建新策略并独立初始化
            new_routing = deepcopy(routing_policy)
            new_sequencing = deepcopy(sequencing_policy)
            new_routing.norm_init()
            new_sequencing.norm_init()

            # 获取参数并存储为元组
            routing_params = get_flatten_params(new_routing)['params']
            sequencing_params = get_flatten_params(new_sequencing)['params']
            self.population.append((routing_params, sequencing_params))

        return self._create_policy_objects()

    def _create_policy_objects(self):
        """将参数列表转换为策略对象列表"""
        perturbations = []

        # 首先添加原始模型对
        base_routing = deepcopy(self.mu_models[0])
        base_sequencing = deepcopy(self.mu_models[1])
        base_routing.set_policy_id(0)
        base_sequencing.set_policy_id(0)
        perturbations.append(agent_policy(
            self.agent_ids,
            (base_routing, base_sequencing)
        ))

        # 添加种群中的个体
        for i, (routing_params, sequencing_params) in enumerate(self.population):
            # 创建路由策略
            routing_policy = deepcopy(self.mu_models[0])
            set_flatten_params(
                routing_params,
                get_flatten_params(routing_policy)['lengths'],
                routing_policy
            )
            routing_policy.set_policy_id(i + 1)

            # 创建排序策略
            sequencing_policy = deepcopy(self.mu_models[1])
            set_flatten_params(
                sequencing_params,
                get_flatten_params(sequencing_policy)['lengths'],
                sequencing_policy
            )
            sequencing_policy.set_policy_id(i + 1)

            perturbations.append(agent_policy(
                self.agent_ids,
                (routing_policy, sequencing_policy)
            ))

        return perturbations

    def crossover(self, parent1, parent2):
        """交叉操作: 随机选择一个策略进行交叉，另一个策略直接交换"""
        routing1, sequencing1 = parent1
        routing2, sequencing2 = parent2

        if random.random() < self.crossover_rate:
            # 随机选择要交叉的策略
            if random.random() < 0.5:
                # 交叉路由策略，交换排序策略
                crossover_point = random.randint(1, len(routing1) - 1)
                new_routing = np.concatenate([
                    routing1[:crossover_point],
                    routing2[crossover_point:]
                ])
                return (new_routing, sequencing2)
            else:
                # 交叉排序策略，交换路由策略
                crossover_point = random.randint(1, len(sequencing1) - 1)
                new_sequencing = np.concatenate([
                    sequencing1[:crossover_point],
                    sequencing2[crossover_point:]
                ])
                return (routing2, new_sequencing)
        return parent1

    def mutate(self, individual):
        """对两个策略分别进行高斯变异"""
        routing, sequencing = individual

        # 变异路由策略
        mask = np.random.random(size=routing.shape) < self.mutation_rate
        noise = np.random.normal(0, self.mutation_scale, size=routing.shape)
        mutated_routing = routing + mask * noise

        # 变异排序策略
        mask = np.random.random(size=sequencing.shape) < self.mutation_rate
        noise = np.random.normal(0, self.mutation_scale, size=sequencing.shape)
        mutated_sequencing = sequencing + mask * noise

        return (mutated_routing, mutated_sequencing)

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
            best_routing, best_sequencing = self.population[best_idx - 1]

            # 更新路由模型
            set_flatten_params(
                best_routing,
                get_flatten_params(self.mu_models[0])['lengths'],
                self.mu_models[0]
            )
            # 更新排序模型
            set_flatten_params(
                best_sequencing,
                get_flatten_params(self.mu_models[1])['lengths'],
                self.mu_models[1]
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
            # 计算两个策略的多样性平均值
            routing_diversity = np.mean([np.std(ind[0]) for ind in self.population])
            sequencing_diversity = np.mean([np.std(ind[1]) for ind in self.population])
            avg_diversity = (routing_diversity + sequencing_diversity) / 2

            self.mutation_rate = max(0.01, min(0.5, 0.1 * (1 + np.tanh(avg_diversity - 1))))

        return self._create_policy_objects(), self.mutation_rate, max(fitness)

    def get_elite_model(self):
        """获取最佳模型对"""
        return self.mu_models