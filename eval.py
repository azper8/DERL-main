"""
强化学习评估主程序 - 用于测试训练好的模型
"""
import yaml
import torch
import heapq
import pickle
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

from builder import build_policy
from env.lib.dataset import CreateDataset
from env.lib.observe import get_observe_R, get_observe_S
from env.lib.event import jobAssignEvent, jobProcessEvent

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning,
                        message='.*numpy.core.numeric.*')


# ============================================================
#  CONFIG：直接改这里
# ============================================================
LOG_PATH = 'logs/RS20260926173842331751'   # 训练日志目录
SEED = 52


# ============================================================
#  工具函数
# ============================================================
def set_seed(seed):
    """设置随机种子以保证可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
#  评估 worker
# ============================================================
def worker_func(arguments):
    envs, RAmodel, ra_ob_rms_mean, ra_ob_rms_std, SAmodel, sa_ob_rms_mean, sa_ob_rms_std = arguments
    total_reward = 0
    total_makespan = 0
    total_violation_rate = 0
    for env in envs:
        env.reset()
        mean_pa = sum(server.pa for server in env.servers) / len(env.servers)
        mean_bandwidth = sum(server.bandwidth for server in env.servers) / len(env.servers)
        for wf in env.workflows:
            wf.calculate_downward_ranks(mean_pa, mean_bandwidth)
            wf.calculate_upward_ranks(mean_pa, mean_bandwidth)
        # 执行episode
        while True:
            # 1. 完成当前时刻所有任务
            while env.event_list and env.event_list[0].time == env.current_time:
                event = heapq.heappop(env.event_list)
                event.trigger(env)

            # 2. 路由选择
            removed_server = []
            if env.ready_job_set:
                select_job = min(env.ready_job_set, key=lambda x: x.workload)
                state = get_observe_R(env, select_job)
                # 归一化观测值
                if ra_ob_rms_mean is not None:
                    state = (state - ra_ob_rms_mean) / ra_ob_rms_std
                action = RAmodel(state, removed=removed_server)
                if action == len(env.servers):
                    env.addEvent(jobAssignEvent(env.workflows[select_job.workflow_id], select_job, env.devices[select_job.sourceDeviceId], env.current_time))
                else:
                    env.addEvent(jobAssignEvent(env.workflows[select_job.workflow_id], select_job, env.servers[action], env.current_time))
                continue

            # 3. 顺序选择
            for device in env.devices:
                if device.isBusy == False and len(device.wait_list) == 1:
                    job = device.wait_list[0]
                    env.addEvent(jobProcessEvent(env.workflows[job.workflow_id], job, device, env.current_time))
                    continue
                elif device.isBusy == False and len(device.wait_list) > 1:
                    state = get_observe_S(env, device)
                    # 归一化观测值
                    if sa_ob_rms_mean is not None:
                        state = (state - sa_ob_rms_mean) / sa_ob_rms_std
                    action = SAmodel(state, removed=[])
                    job = device.wait_list[action]
                    env.addEvent(jobProcessEvent(env.workflows[job.workflow_id], job, device, env.current_time))
                    continue

            for server in env.servers:
                if server.available and server.isBusy == False and len(server.wait_list) == 1:
                    job = server.wait_list[0]
                    env.addEvent(jobProcessEvent(env.workflows[job.workflow_id], job, server, env.current_time))
                    continue
                elif server.available and server.isBusy == False and len(server.wait_list) > 1:
                    state = get_observe_S(env, server)
                    # 归一化观测值
                    if sa_ob_rms_mean is not None:
                        state = (state - sa_ob_rms_mean) / sa_ob_rms_std
                    action = SAmodel(state, removed=[])
                    job = server.wait_list[action]
                    env.addEvent(jobProcessEvent(env.workflows[job.workflow_id], job, server, env.current_time))
                    continue

            # 4. 检查任务是否全部完成（区分出口 / 非出口）
            if env.event_list == []:
                for wf in env.workflows:
                    for job in wf.jobs:
                        is_exit = (len(list(wf.graph.successors(job.id))) == 0)
                        if is_exit:
                            if job.state != "已完成":
                                print(f"工作流{wf.id} 出口任务{job.id}未完成，状态: {job.state}")
                        else:
                            if job.state not in ["执行完成", "已完成"]:
                                print(f"工作流{wf.id} 任务{job.id}未完成，状态: {job.state}")
                break

            # 5. 更新时间
            if env.current_time != env.event_list[0].time:
                env.current_time = env.event_list[0].time

        # 计算 makespan（用封装方法，带 None 保护）
        makespan = 0
        for wf in env.workflows:
            wf_finish = wf.getWorkflowFinishTime()
            if wf_finish > makespan:
                makespan = wf_finish

        count = 0
        for wf in env.workflows:
            if wf.getWorkflowFinishTime() > wf.deadline:
                count += 1

        total_reward += -makespan * (count + 1)
        total_makespan += makespan
        total_violation_rate += count / len(env.workflows)
        del env

    rewards_mean = total_reward / len(envs)
    makespan_mean = total_makespan / len(envs)
    violation_rate_mean = total_violation_rate / len(envs)

    return {'rewards': rewards_mean,
            'makespan': makespan_mean,
            'violation_rate': violation_rate_mean}


# ============================================================
#  主流程
# ============================================================
def main():
    """主评估函数"""
    set_seed(SEED)
    # 1. 准备日志目录
    log_path = LOG_PATH
    with open(log_path + '/profile_test.yaml', 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()
    dataset = CreateDataset(config["env"])
    envs = dataset.create_test_envs()

    for fr in tqdm(np.arange(10, 310, 10, dtype=int)):
        path_r = f'{log_path}/saved_models/ep_{fr}_r.pt'
        path_s = f'{log_path}/saved_models/ep_{fr}_s.pt'
        ob_path_r = f'{log_path}/saved_models/ob_rms_{fr}_r.pickle'
        ob_path_s = f'{log_path}/saved_models/ob_rms_{fr}_s.pickle'
        RAPolicy = build_policy(config["policy"])
        RAPolicy.load_state_dict(torch.load(path_r))
        with open(ob_path_r, "rb") as f:
            ra_ob_rms = pickle.load(f)
            ra_ob_rms_mean = ra_ob_rms[:int(0.5 * len(ra_ob_rms))]
            ra_ob_rms_std = ra_ob_rms[int(0.5 * len(ra_ob_rms)):]
        SAPolicy = build_policy(config["policy"])
        SAPolicy.load_state_dict(torch.load(path_s))
        with open(ob_path_s, "rb") as f:
            sa_ob_rms = pickle.load(f)
            sa_ob_rms_mean = sa_ob_rms[:int(0.5 * len(sa_ob_rms))]
            sa_ob_rms_std = sa_ob_rms[int(0.5 * len(sa_ob_rms)):]

        arguments = (envs, RAPolicy, ra_ob_rms_mean, ra_ob_rms_std, SAPolicy, sa_ob_rms_mean, sa_ob_rms_std)
        results = worker_func(arguments)
        results_df = pd.DataFrame(results, index=[0])
        print(results_df)

        dir_test = log_path + "/test_performance"
        test_size = config['env']['wf_size']
        test_mobile_num = config['env']['mobile_num']
        results_df.to_csv(dir_test + "/testing_record_" + str(test_size) + '-' + str(test_mobile_num) + ".csv", index=False, header=False, mode='a')


if __name__ == "__main__":
    main()  # 启动评估