"""
强化学习训练主程序
"""
import time
import torch
import random
import numpy as np
from builder import Builder
from config.base_config import BaseConfig


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # 1. 加载基础配置
    baseconfig = BaseConfig()
    RAorSA = 'RASA'
    if RAorSA == 'SA':
        path1 = 'logs/workflow_scheduling/RA20251228131822921588/saved_models/ep_190.pt'
        path2 = 'logs/workflow_scheduling/RA20251228131822921588/saved_models/ob_rms_190.pickle'
        path3 = 'logs/workflow_scheduling/RS20251206141329431810/saved_models/test_env.pkl'
    elif RAorSA == 'RA':
        path1 = None
        path2 = None
        path3 = 'logs/workflow_scheduling/RS20251206141329431810/saved_models/test_env.pkl'
    else:
        path1 = None
        path2 = None
        path3 = None

    # 2. 设置全局随机种子
    set_seed(baseconfig.config["yaml-config"]['env']['seed'])

    # 3. 构建并训练RL系统(RA模式下path分别是RA的模型和rms,RASA模式下path分别是RA和SA的模型)
    Builder(baseconfig, RAorSA=RAorSA, path1=path1, path2=path2, path3=path3).build().train()


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    execution_time = end_time - start_time
    with open('training_time.txt', 'a') as f:
        f.write(f'程序执行时间: {execution_time:.2f} 秒\n')

