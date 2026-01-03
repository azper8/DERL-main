import numpy as np
from copy import deepcopy
from utils.policy_dict import agent_policy
from utils.torch_util import get_flatten_params, set_flatten_params
from policy.rl_model import RLPolicy


class DEOptimizer:
    """差分进化优化器"""

    def __init__(self, config):
        """初始化差分进化参数"""
        self.config = config
        self.crossover_rate = config.get("crossover_rate")
        self.zoom_factor1 = config.get("zoom_factor1")
        self.zoom_factor2 = config.get("zoom_factor2")
        self.population_size = config.get("population_size") + 1
        self.strategy = config.get("de_strategy", "rand/1/bin")  # 差分进化策略
        self.agent_ids = None

        # 支持多种差分进化策略
        self.SUPPORTED_STRATEGIES = [
            "rand/1/bin", "best/1/bin", "rand/2/bin", "best/2/bin",
            "rand/1/exp", "current-to-rand/1", "current-to-best/1"
        ]

        # 运行时变量
        self.population = []
        self.best_individual = None

    def init_population(self, policy_tuple, env):
        self.agent_ids = env.get_agent_ids()
        self.best_individual = agent_policy(self.agent_ids, policy_tuple)
        self.population.append(agent_policy(self.agent_ids, policy_tuple))
        config = {
                  "name": 'rl',
                  "state_num": 8,
                  "action_num": 1,
                  "d_model": 16,
                  "discrete_action": True,
                  "add_gru": False,
                  "action_type": 'greedy'
                  }
        for i in range(self.population_size):
            r = RLPolicy(config)
            r.set_policy_id(i)
            s = RLPolicy(config)
            s.set_policy_id(i)
            self.population.append(agent_policy(self.agent_ids, (r, s)))
        return self.population

    @staticmethod
    def _extract_parameters(policy_dict):
        """从agent_policy中提取参数"""
        agent_id = list(policy_dict.keys())[0]
        routing_policy, sequencing_policy = policy_dict[agent_id]

        routing_params = get_flatten_params(routing_policy)['params']
        sequencing_params = get_flatten_params(sequencing_policy)['params']

        return routing_params, sequencing_params, routing_policy, sequencing_policy

    def _create_agent_policy(self, routing_policy, sequencing_policy, policy_id=0):
        """创建agent_policy"""
        new_routing = deepcopy(routing_policy)
        new_sequencing = deepcopy(sequencing_policy)

        if policy_id > 0:
            new_routing.set_policy_id(policy_id)
            new_sequencing.set_policy_id(policy_id)

        return agent_policy(self.agent_ids, (new_routing, new_sequencing))

    def next_population(self, results, generation=None):
        """使用差分进化生成下一代种群"""
        fitness_scores = np.array(results['rewards'].tolist())
        makespans = np.array(results['makespan']).tolist()
        overrates = np.array(results['violation_rate'].tolist())

        best_reward_idx = np.argmax(fitness_scores)
        best_reward = fitness_scores[best_reward_idx]
        best_makespan = makespans[best_reward_idx]
        best_overrate = overrates[best_reward_idx]
        self.best_individual = self.population[best_reward_idx]

        # 提取参数种群
        param_population = []
        policy_templates = []

        for policy_dict in self.population:
            routing_params, sequencing_params, routing_policy, sequencing_policy = self._extract_parameters(policy_dict)
            param_population.append((routing_params, sequencing_params))
            policy_templates.append((routing_policy, sequencing_policy))

        # 执行差分进化
        new_param_population = self._differential_evolution(
            param_population, fitness_scores, generation
        )

        # 创建新的策略种群并更新self.population
        new_population = []
        for i, (routing_params, sequencing_params) in enumerate(new_param_population):
            # 使用对应的策略作为模板
            routing_policy_template, sequencing_policy_template = policy_templates[i]

            # 设置参数
            set_flatten_params(routing_params, get_flatten_params(routing_policy_template)['lengths'], routing_policy_template)
            set_flatten_params(sequencing_params, get_flatten_params(sequencing_policy_template)['lengths'], sequencing_policy_template)

            # 创建新策略（重用原对象，更新ID）
            routing_policy_template.set_policy_id(i + 1)
            sequencing_policy_template.set_policy_id(i + 1)

            new_policy = agent_policy(self.agent_ids, (routing_policy_template, sequencing_policy_template))
            new_population.append(new_policy)

        # 更新种群
        self.population = new_population

        return self.population, best_reward, best_makespan, best_overrate, self.best_individual

    def _differential_evolution(self, population, fitness, generation):
        """执行差分进化操作"""
        new_population = []
        population_size = len(population)

        for i in range(population_size):
            # 根据策略选择变异方法
            if self.strategy == "rand/1/bin":
                mutant = self._rand_1_bin_mutation(population, i, fitness)
            elif self.strategy == "best/1/bin":
                mutant = self._best_1_bin_mutation(population, i, fitness)
            elif self.strategy == "rand/2/bin":
                mutant = self._rand_2_bin_mutation(population, i, fitness)
            elif self.strategy == "best/2/bin":
                mutant = self._best_2_bin_mutation(population, i, fitness)
            elif self.strategy == "rand/1/exp":
                mutant = self._rand_1_exp_mutation(population, i, fitness)
            elif self.strategy == "current-to-rand/1":
                mutant = self._current_to_rand_1_mutation(population, i, fitness)
            elif self.strategy == "current-to-best/1":
                mutant = self._current_to_best_1_mutation(population, i, fitness)
            else:
                mutant = self._rand_1_bin_mutation(population, i, fitness)

            # 边界处理
            mutant = self._boundary_handling(mutant)

            new_population.append(mutant)

        return new_population

    def _rand_1_bin_mutation(self, population, idx, fitness):
        """rand/1/bin策略：随机选择3个不同个体"""
        indices = [i for i in range(len(population)) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)

        # 获取参数
        target_routing, target_sequencing = population[idx]
        r1_routing, r1_sequencing = population[a]
        r2_routing, r2_sequencing = population[b]
        r3_routing, r3_sequencing = population[c]

        # 变异：V = X_r1 + F * (X_r2 - X_r3)
        mutant_routing = r1_routing + self.zoom_factor1 * (r2_routing - r3_routing)
        mutant_sequencing = r1_sequencing + self.zoom_factor1 * (r2_sequencing - r3_sequencing)

        # 二项交叉
        trial_routing, trial_sequencing = self._binomial_crossover(
            target_routing, target_sequencing,
            mutant_routing, mutant_sequencing
        )

        return (trial_routing, trial_sequencing)

    def _best_1_bin_mutation(self, population, idx, fitness):
        """best/1/bin策略：选择最佳个体"""
        best_idx = np.argmax(fitness)
        indices = [i for i in range(len(population)) if i != idx and i != best_idx]
        a, b = np.random.choice(indices, 2, replace=False)

        # 获取参数
        target_routing, target_sequencing = population[idx]
        best_routing, best_sequencing = population[best_idx]
        r1_routing, r1_sequencing = population[a]
        r2_routing, r2_sequencing = population[b]

        # 变异：V = X_best + F * (X_r1 - X_r2)
        mutant_routing = best_routing + self.zoom_factor * (r1_routing - r2_routing)
        mutant_sequencing = best_sequencing + self.zoom_factor * (r1_sequencing - r2_sequencing)

        # 二项交叉
        trial_routing, trial_sequencing = self._binomial_crossover(
            target_routing, target_sequencing,
            mutant_routing, mutant_sequencing
        )

        return (trial_routing, trial_sequencing)

    def _rand_2_bin_mutation(self, population, idx, fitness):
        """rand/2/bin策略：随机选择5个不同个体"""
        indices = [i for i in range(len(population)) if i != idx]
        a, b, c, d, e = np.random.choice(indices, 5, replace=False)

        # 获取参数
        target_routing, target_sequencing = population[idx]
        r1_routing, r1_sequencing = population[a]
        r2_routing, r2_sequencing = population[b]
        r3_routing, r3_sequencing = population[c]
        r4_routing, r4_sequencing = population[d]
        r5_routing, r5_sequencing = population[e]

        # 变异：V = X_r1 + F1 * (X_r2 - X_r3) + F2 * (X_r4 - X_r5)
        mutant_routing = (r1_routing +
                          self.zoom_factor * (r2_routing - r3_routing) +
                          0.5 * self.zoom_factor * (r4_routing - r5_routing))
        mutant_sequencing = (r1_sequencing +
                             self.zoom_factor * (r2_sequencing - r3_sequencing) +
                             0.5 * self.zoom_factor * (r4_sequencing - r5_sequencing))

        # 二项交叉
        trial_routing, trial_sequencing = self._binomial_crossover(
            target_routing, target_sequencing,
            mutant_routing, mutant_sequencing
        )

        return (trial_routing, trial_sequencing)

    def _current_to_rand_1_mutation(self, population, idx, fitness):
        """current-to-rand/1策略"""
        indices = [i for i in range(len(population)) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)

        # 获取参数
        target_routing, target_sequencing = population[idx]
        r1_routing, r1_sequencing = population[a]
        r2_routing, r2_sequencing = population[b]
        r3_routing, r3_sequencing = population[c]

        # 当前到随机变异：U = X + K * (X_r1 - X) + F * (X_r2 - X_r3)
        K = 0.5  # 混合系数
        mutant_routing = (target_routing +
                          K * (r1_routing - target_routing) +
                          self.zoom_factor * (r2_routing - r3_routing))
        mutant_sequencing = (target_sequencing +
                             K * (r1_sequencing - target_sequencing) +
                             self.zoom_factor * (r2_sequencing - r3_sequencing))

        return (mutant_routing, mutant_sequencing)

    def _current_to_best_1_mutation(self, population, idx, fitness):
        """current-to-best/1策略"""
        best_idx = np.argmax(fitness)
        indices = [i for i in range(len(population)) if i != idx and i != best_idx]
        a, b = np.random.choice(indices, 2, replace=False)

        # 获取参数
        target_routing, target_sequencing = population[idx]
        best_routing, best_sequencing = population[best_idx]
        r1_routing, r1_sequencing = population[a]
        r2_routing, r2_sequencing = population[b]

        # 当前到最佳变异：V = X + F1 * (X_best - X) + F2 * (X_r1 - X_r2)
        mutant_routing = (target_routing +
                          self.zoom_factor1 * (best_routing - target_routing) +
                          self.zoom_factor2 * (r1_routing - r2_routing))
        mutant_sequencing = (target_sequencing +
                             self.zoom_factor1 * (best_sequencing - target_sequencing) +
                             self.zoom_factor2 * (r1_sequencing - r2_sequencing))

        # 二项交叉
        trial_routing, trial_sequencing = self._binomial_crossover(
            target_routing, target_sequencing,
            mutant_routing, mutant_sequencing
        )

        return (trial_routing, trial_sequencing)

    def _rand_1_exp_mutation(self, population, idx, fitness):
        """rand/1/exp策略：指数交叉"""
        indices = [i for i in range(len(population)) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)

        # 获取参数
        target_routing, target_sequencing = population[idx]
        r1_routing, r1_sequencing = population[a]
        r2_routing, r2_sequencing = population[b]
        r3_routing, r3_sequencing = population[c]

        # 变异
        mutant_routing = r1_routing + self.zoom_factor * (r2_routing - r3_routing)
        mutant_sequencing = r1_sequencing + self.zoom_factor * (r2_sequencing - r3_sequencing)

        # 指数交叉
        trial_routing, trial_sequencing = self._exponential_crossover(
            target_routing, target_sequencing,
            mutant_routing, mutant_sequencing
        )

        return (trial_routing, trial_sequencing)

    def _binomial_crossover(self, target_routing, target_sequencing,
                            mutant_routing, mutant_sequencing):
        """二项交叉（binomial crossover）"""
        # 生成交叉掩码
        routing_mask = np.random.random(size=target_routing.shape) < self.crossover_rate
        sequencing_mask = np.random.random(size=target_sequencing.shape) < self.crossover_rate

        # 确保至少有一个基因被交叉
        routing_idx = np.random.randint(0, len(target_routing))
        sequencing_idx = np.random.randint(0, len(target_sequencing))
        routing_mask[routing_idx] = True
        sequencing_mask[sequencing_idx] = True

        # 执行交叉
        trial_routing = np.where(routing_mask, mutant_routing, target_routing)
        trial_sequencing = np.where(sequencing_mask, mutant_sequencing, target_sequencing)

        return trial_routing, trial_sequencing

    def _exponential_crossover(self, target_routing, target_sequencing,
                               mutant_routing, mutant_sequencing):
        """指数交叉（exponential crossover）"""
        n_routing = len(target_routing)
        n_sequencing = len(target_sequencing)

        # 随机选择起始点
        start_routing = np.random.randint(0, n_routing)
        start_sequencing = np.random.randint(0, n_sequencing)

        # 创建交叉掩码
        routing_mask = np.zeros(n_routing, dtype=bool)
        sequencing_mask = np.zeros(n_sequencing, dtype=bool)

        # 生成指数交叉模式
        L_routing = 0
        L_sequencing = 0
        while (np.random.rand() < self.crossover_rate and L_routing < n_routing):
            routing_mask[(start_routing + L_routing) % n_routing] = True
            L_routing += 1

        while (np.random.rand() < self.crossover_rate and L_sequencing < n_sequencing):
            sequencing_mask[(start_sequencing + L_sequencing) % n_sequencing] = True
            L_sequencing += 1

        # 执行交叉
        trial_routing = np.where(routing_mask, mutant_routing, target_routing)
        trial_sequencing = np.where(sequencing_mask, mutant_sequencing, target_sequencing)

        return trial_routing, trial_sequencing

    @staticmethod
    def _boundary_handling(individual):
        """边界处理（如果需要约束参数范围）"""
        routing, sequencing = individual

        # 这里可以添加参数边界约束
        # 例如：routing = np.clip(routing, -10, 10)
        # sequencing = np.clip(sequencing, -10, 10)

        return (routing, sequencing)

    def get_elite_individual(self):
        """获取精英模型"""
        return self.best_individual

    def get_elite_model(self):
        _, _, routing_policy, sequencing_policy = self._extract_parameters(self.best_individual)
        return (routing_policy, sequencing_policy)