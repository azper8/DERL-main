import torch
import numpy as np
from copy import deepcopy
from optim.base_optim import BaseOptim
from utils.policy_dict import agent_policy
from utils.torch_util import get_flatten_params, set_flatten_params


class ESOptimizer(BaseOptim):

    def __init__(self, config):
        super().__init__()
        self._init_config(config)
        self._init_variables()

    def _init_config(self, config):
        """初始化配置参数"""
        self.config = config
        self.sigma = config.get("sigma_init")
        self.sigma_decay = config.get("sigma_decay")
        self.generations = config.get("generation_num")
        self.learning_rate = config.get("learning_rate")
        self.population_size = config.get("population_size")

    def _init_variables(self):
        """初始化运行时变量"""
        self.layer_weights = [0.0, 0.0, 1.0]
        self.population = None
        self.epsilons = []
        self.mu_model = None
        self.optimizer = None
        self.agent_ids = None
        self.routing_bounds = None
        self.sequencing_bounds = None

    def init_population(self, policy_tuple, env):
        """初始化种群"""
        self.agent_ids = env.get_agent_ids()
        routing_policy, sequencing_policy = policy_tuple
        self.mu_model = (routing_policy, sequencing_policy)

        # 获取参数结构
        routing_params = get_flatten_params(routing_policy)
        sequencing_params = get_flatten_params(sequencing_policy)

        # 计算各层参数边界
        self.routing_bounds = self._get_layer_boundaries(routing_policy)
        self.sequencing_bounds = self._get_layer_boundaries(sequencing_policy)

        # 初始化优化器
        joint_params = np.concatenate([routing_params['params'], sequencing_params['params']])
        self.optimizer = torch.optim.Adam(
            [torch.tensor(joint_params, requires_grad=True, dtype=torch.float64)],
            lr=self.learning_rate
        )

        # 生成初始种群
        self.population = self._generate_perturbations()
        return self.population

    @staticmethod
    def _get_layer_boundaries(policy):
        """计算三层网络的参数边界"""
        state_dim = policy.fc1.in_features
        hidden_dim = policy.fc1.out_features
        action_dim = policy.fc3.out_features

        fc1_weight_size = state_dim * hidden_dim
        fc1_bias_size = hidden_dim
        fc1_total = fc1_weight_size + fc1_bias_size

        fc2_weight_size = hidden_dim * hidden_dim
        fc2_bias_size = hidden_dim
        fc2_total = fc2_weight_size + fc2_bias_size

        fc3_weight_size = hidden_dim * action_dim
        fc3_bias_size = action_dim
        fc3_total = fc3_weight_size + fc3_bias_size

        return [
            (0, fc1_total),
            (fc1_total, fc1_total + fc2_total),
            (fc1_total + fc2_total, fc1_total + fc2_total + fc3_total)
        ]

    def _generate_perturbations(self):
        """生成带分层噪声的种群（包含扰动对）"""
        perturbations = []
        self.epsilons = []

        # 主策略（无噪声）
        perturbations.append(agent_policy(self.agent_ids, self.mu_model))
        self.epsilons.append(np.zeros(sum(end - start for start, end in self.routing_bounds) * 2))

        # 生成扰动对（正负对称）
        for i in range(1, self.population_size // 2 + 1):
            # 生成基础噪声
            epsilon = self._generate_layer_noise(self.sigma)

            # 扰动对1: +epsilon
            routing_copy1 = deepcopy(self.mu_model[0])
            sequencing_copy1 = deepcopy(self.mu_model[1])
            routing_copy1.set_policy_id(2 * i - 1)
            sequencing_copy1.set_policy_id(2 * i - 1)

            self._apply_layer_perturbation(routing_copy1, sequencing_copy1, epsilon)
            perturbations.append(agent_policy(self.agent_ids, (routing_copy1, sequencing_copy1)))
            self.epsilons.append(epsilon)

            # 扰动对2: -epsilon
            routing_copy2 = deepcopy(self.mu_model[0])
            sequencing_copy2 = deepcopy(self.mu_model[1])
            routing_copy2.set_policy_id(2 * i)
            sequencing_copy2.set_policy_id(2 * i)

            self._apply_layer_perturbation(routing_copy2, sequencing_copy2, -epsilon)
            perturbations.append(agent_policy(self.agent_ids, (routing_copy2, sequencing_copy2)))
            self.epsilons.append(-epsilon)

        # 如果种群大小为奇数，添加一个额外的随机扰动
        if self.population_size % 2 == 1:
            routing_copy = deepcopy(self.mu_model[0])
            sequencing_copy = deepcopy(self.mu_model[1])
            routing_copy.set_policy_id(self.population_size)
            sequencing_copy.set_policy_id(self.population_size)

            epsilon = self._generate_layer_noise(self.sigma)
            self.epsilons.append(epsilon)
            self._apply_layer_perturbation(routing_copy, sequencing_copy, epsilon)
            perturbations.append(agent_policy(self.agent_ids, (routing_copy, sequencing_copy)))

        return perturbations

    def _generate_layer_noise(self, sigma, weights=None):
        """分层生成噪声向量"""
        total_params = sum(end - start for start, end in self.routing_bounds) * 2
        epsilon = np.zeros(total_params)

        if weights is not None:
            for i, (weight, (r_start, r_end), (s_start, s_end)) in enumerate(zip(
                    weights, self.routing_bounds, self.sequencing_bounds)):
                actual_r_start = r_start
                actual_r_end = r_end
                actual_s_start = s_start + sum(end - start for start, end in self.routing_bounds)
                actual_s_end = s_end + sum(end - start for start, end in self.routing_bounds)

                layer_size = r_end - r_start
                epsilon[actual_r_start:actual_r_end] = np.random.normal(scale=weight * sigma, size=layer_size)
                epsilon[actual_s_start:actual_s_end] = np.random.normal(scale=weight * sigma, size=layer_size)
        else:
            for i, ((r_start, r_end), (s_start, s_end)) in enumerate(zip(
                    self.routing_bounds, self.sequencing_bounds)):
                actual_r_start = r_start
                actual_r_end = r_end
                actual_s_start = s_start + sum(end - start for start, end in self.routing_bounds)
                actual_s_end = s_end + sum(end - start for start, end in self.routing_bounds)

                layer_size = r_end - r_start
                epsilon[actual_r_start:actual_r_end] = np.random.normal(scale=sigma, size=layer_size)
                epsilon[actual_s_start:actual_s_end] = np.random.normal(scale=sigma, size=layer_size)

        return epsilon

    @staticmethod
    def _apply_layer_perturbation(routing_policy, sequencing_policy, epsilon):
        """应用分层噪声到策略"""
        routing_params = get_flatten_params(routing_policy)
        routing_params['params'] += epsilon[:len(routing_params['params'])]
        set_flatten_params(routing_params['params'], routing_params['lengths'], routing_policy)

        sequencing_params = get_flatten_params(sequencing_policy)
        sequencing_params['params'] += epsilon[len(routing_params['params']):]
        set_flatten_params(sequencing_params['params'], sequencing_params['lengths'], sequencing_policy)

    def next_population(self, results, generation=None):
        """生成下一代种群"""
        rewards = np.array(results['rewards'].tolist())
        makespans = np.array(results['makespan']).tolist()
        overrates = np.array(results['violation_rate'].tolist())

        best_reward_idx = np.argmax(rewards)
        best_reward = rewards[best_reward_idx]
        best_makespan = makespans[best_reward_idx]
        best_overrate = overrates[best_reward_idx]
        best_individual = self.population[best_reward_idx]

        rewards_norm = self._normalize_rewards(rewards)
        grad = self._compute_gradients(rewards_norm)
        self._update_parameters(grad)
        self.population = self._generate_perturbations()
        self._adapt_parameters(generation)

        return self.population, self.sigma, best_reward, best_makespan, best_overrate, best_individual

    @staticmethod
    def _normalize_rewards(rewards):
        """奖励中心化处理"""
        if len(rewards) <= 1:
            return np.zeros_like(rewards)

        ranks = np.argsort(np.argsort(rewards))
        return (ranks / (len(rewards) - 1)) - 0.5

    def _compute_gradients(self, rewards):
        """计算基本梯度"""
        update_factor = -1.0 / (self.population_size * self.sigma)
        return np.sum([e * r for e, r in zip(self.epsilons[1:], rewards[1:])], axis=0) * update_factor

    def _update_parameters(self, grad):
        """更新模型参数"""
        self.optimizer.zero_grad()
        self.optimizer.param_groups[0]['params'][0].grad = torch.tensor(grad, dtype=torch.float64)
        self.optimizer.step()

        # 同步参数到模型
        joint_params = self.optimizer.param_groups[0]['params'][0].detach().numpy()
        routing_policy, sequencing_policy = self.mu_model

        routing_params = joint_params[:sum(end - start for start, end in self.routing_bounds)]
        set_flatten_params(routing_params, get_flatten_params(routing_policy)['lengths'], routing_policy)

        sequencing_params = joint_params[sum(end - start for start, end in self.routing_bounds):]
        set_flatten_params(sequencing_params, get_flatten_params(sequencing_policy)['lengths'], sequencing_policy)

    def _adapt_parameters(self, generation):
        """自适应参数调整"""
        if self.sigma >= 0.01:
            self.sigma *= self.sigma_decay

    def get_elite_model(self):
        """获取精英模型"""
        return self.mu_model

    def get_elite_individual(self):
        routing_copy = deepcopy(self.mu_model[0])
        sequencing_copy = deepcopy(self.mu_model[1])
        return agent_policy(self.agent_ids, (routing_copy, sequencing_copy))

