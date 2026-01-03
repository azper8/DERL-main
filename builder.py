import torch
import pickle


class Builder:

    def __init__(self, baseconfig, RAorSA, path1=None, path2=None, path3=None):

        self.config = baseconfig
        self.RAorSA = RAorSA
        self.path1 = path1
        self.path2 = path2
        self.path3 = path3

    def build(self):

        # 1. 构建数据集
        dataset = build_dataset(self.config.config)

        # 2. 构建策略和优化器
        policy = build_policy(self.config.config['yaml-config']["policy"])
        optim = build_optim(self.config.config['yaml-config']["optim"])

        # 3. 获取RAPolicy
        if self.RAorSA == 'RA':
            RAPolicy = None
            ob_rms_mean = None
            ob_rms_std = None
            with open(self.path3, 'rb') as f:
                test_env = pickle.load(f)
        elif self.RAorSA == 'SA':
            RAPolicy = build_policy(self.config.config['yaml-config']["policy"])
            RAPolicy.load_state_dict(torch.load(self.path1, weights_only=True))
            with open(self.path2, "rb") as f:
                ob_rms = pickle.load(f)
                ob_rms_mean = ob_rms[:int(0.5 * len(ob_rms))]
                ob_rms_std = ob_rms[int(0.5 * len(ob_rms)):]
            with open(self.path3, 'rb') as f:
                test_env = pickle.load(f)
        elif self.RAorSA == 'RASA':
            policy2 = build_policy(self.config.config['yaml-config']["policy"])
        else:
            policy2 = build_policy(self.config.config['yaml-config']["policy"])
            optim2 = build_optim(self.config.config['yaml-config']["optim"])

        # 4. 返回组装好的RL系统
        if self.RAorSA == 'RA' or self.RAorSA == 'SA':
            from assembly.assemble_sprl import AssembleRL
            return AssembleRL(self.config, dataset, test_env, policy, optim, self.RAorSA, RAPolicy, ob_rms_mean, ob_rms_std)
        elif self.RAorSA == 'RASA':
            from assembly.assemble_mprl import AssembleRL
            return AssembleRL(self.config, dataset, policy, policy2, optim)
        else:
            from assembly.assemble_strl import AssembleRL
            return AssembleRL(self.config, dataset, policy, policy2, optim, optim2)


def build_dataset(config):
    env_name = config['yaml-config']["env"]["name"]
    config['yaml-config']['env']['evalNum'] = config['runtime-config']['eval_ep_num']
    if env_name == "WorkflowScheduling":
        from env.workflow_scheduling.lib.createDataset import CreateDataset
        return CreateDataset(config['yaml-config']["env"])
    else:
        raise AssertionError(f"不支持的环境: {env_name}, 请在yaml中指定支持的环境")


def build_policy(config):
    model_name = config["name"]
    if model_name == "rl":
        from policy.rl_model import RLPolicy
        return RLPolicy(config)
    else:
        raise AssertionError(f"不支持的模型: {model_name}, 请在yaml中指定支持的模型")


def build_optim(config):
    optim_name = config["name"]
    if optim_name == "es":
        from optim.evo.es import ESOpenAI
        return ESOpenAI(config)
    elif optim_name == "mpes":
        from optim.evo.mpes import ESOptimizer
        return ESOptimizer(config)
    else:
        raise AssertionError(f"不支持的优化器: {optim_name}, 请在yaml中指定支持的优化器")
