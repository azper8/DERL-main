# base_config.py
from abc import *
import argparse
import yaml


class BaseConfig(metaclass=ABCMeta):
    """基础配置类，用于解析命令行参数和YAML配置文件"""

    def __init__(self, *args):
        # 初始化参数解析器
        parser = argparse.ArgumentParser(description='参数配置')

        # 添加命令行参数
        parser.add_argument('--config', type=str, default='config/workflow_scheduling.yaml', help='环境、策略和优化器的配置文件路径')
        parser.add_argument('--processor_num', type=int, default=4, help='指定多进程处理的处理器数量')
        parser.add_argument('--eval_ep_num', type=int, default=1, help='每次迭代的评估次数')

        # 日志相关设置
        parser.add_argument("--log", default=True, action="store_true", help="是否启用日志记录")
        parser.add_argument('--save_model_freq', type=int, default=10, help='每隔多少迭代保存一次模型')

        # 用于覆盖YAML中的常见值
        parser.add_argument('--seed', type=int, default=None, help='覆盖YAML中的随机种子值')
        parser.add_argument('--reward', type=int, default=None, help='选择奖励选项')
        parser.add_argument('--sigma_init', type=float, default=None, help='噪声标准差的初始值')
        parser.add_argument('--learning_rate', type=float, default=None, help='覆盖YAML中的学习率')
        parser.add_argument('--reinforce_learning_rate', type=float, default=None, help='覆盖YAML中的强化学习率')

        # 解析命令行参数
        args = parser.parse_args()

        # 加载YAML配置文件
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            f.close()

            # 如果命令行参数不为None，则覆盖YAML中的对应值
            if args.seed is not None:
                config['env']['seed'] = args.seed
            if args.reward is not None:
                config['env']['reward'] = args.reward
            if args.sigma_init is not None:
                config['optim']['sigma_init'] = args.sigma_init
            if args.learning_rate is not None:
                config['optim']['learning_rate'] = args.learning_rate
            if args.reinforce_learning_rate is not None:
                config['optim']['reinforce_learning_rate'] = args.reinforce_learning_rate

        # 存储配置信息
        self.config = {}
        self.config["runtime-config"] = vars(args)  # 运行时配置（命令行参数）
        self.config["yaml-config"] = config  # YAML配置
