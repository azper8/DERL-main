# eval_config.py
from abc import *
import argparse
import yaml


class EvalConfig(metaclass=ABCMeta):
    """评估配置类，用于测试阶段的参数配置"""

    def __init__(self, *args):
        fr, log_path = args  # fr表示模型保存的迭代次数，log_path是日志路径

        # 初始化参数解析器
        parser = argparse.ArgumentParser()

        # 添加命令行参数
        parser.add_argument("--config", type=str, default=f"{log_path}/profile_test.yaml", help='配置文件路径')
        parser.add_argument("--policy_path", type=str, default=f'{log_path}/saved_models/ep_{fr}.pt', help='保存的模型路径')
        parser.add_argument("--rms_path", type=str, default=f'{log_path}/saved_models/ob_rms_{fr}.pickle', help='保存的运行时均值和标准差路径')
        parser.add_argument("--p1_path",type=str,default=f'{log_path}/saved_models/ep_{fr}_r.pt')
        parser.add_argument("--r1_path",type=str,default=f'{log_path}/saved_models/ob_rms_{fr}_r.pickle')
        parser.add_argument("--p2_path", type=str, default=f'{log_path}/saved_models/ep_{fr}_s.pt')
        parser.add_argument("--r2_path", type=str, default=f'{log_path}/saved_models/ob_rms_{fr}_s.pickle')
        parser.add_argument('--eval_ep_num', type=int, default=1, help='每次迭代的评估次数')
        parser.add_argument('--save_model_freq', type=int, default=10, help='每隔多少迭代保存一次模型')
        parser.add_argument('--processor_num', type=int, default=1, help='测试模型时处理器数量')
        parser.add_argument("--log", default=True, action="store_true", help="是否启用日志记录")
        parser.add_argument('--seed', type=int, default=None, help='随机数生成种子')

        # 解析命令行参数
        args = parser.parse_args()

        # 加载YAML配置文件
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            f.close()

            # 覆盖YAML中的随机种子
            if args.seed is not None:
                config['env']['seed'] = args.seed

        # 存储配置信息
        self.config = {}
        self.config["runtime-config"] = vars(args)  # 运行时配置
        self.config["yaml-config"] = config  # YAML配置
