import os
import time
import yaml
import heapq
import torch
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime
import multiprocessing as mp

from assembly.base_assemble import BaseAssemble
from utils.running_mean_std import RunningMeanStd
from utils.policy_dict import agent_policy
from env.workflow_scheduling.lib.observe import get_observe_R, get_observe_S
from env.workflow_scheduling.lib.event import jobAssignEvent, jobProcessEvent
from env.workflow_scheduling.lib.llh import *

running_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AssembleRL(BaseAssemble):
    """强化学习组装类，继承自BaseAssemble，用于训练和评估强化学习策略。"""

    def __init__(self, config, dataset, test_env, policy, optim, RAorSA, RAPolicy=None, rms_mean=None, rms_std=None):
        super(AssembleRL, self).__init__()
        self.config = config
        self.dataset = dataset
        self.dataset = dataset
        self.env = dataset.create_train_env()
        self.env_test = test_env
        self.RAPolicy = RAPolicy
        self.policy = policy
        self.optim = optim
        self.RAorSA = RAorSA
        self.ra_ob_rms_mean = rms_mean
        self.ra_ob_rms_std = rms_std

        # 运行时的均值标准差设置,用于归一化观测值
        self.running_mstd = self.config.config['yaml-config']["optim"]['input_running_mean_std']
        if self.running_mstd:
            self.ob_rms = RunningMeanStd(shape=self.env.observation_space.shape)
            self.ob_rms_mean = self.ob_rms.mean
            self.ob_rms_std = np.sqrt(self.ob_rms.var)
        else:
            self.ob_rms = None
            self.ob_rms_mean = None
            self.ob_rms_std = None

        self.generation_num = self.config.config['yaml-config']['optim']['generation_num']
        self.population_size = self.config.config['yaml-config']['optim']['population_size']
        self.processor_num = self.config.config['runtime-config']['processor_num']
        self.log = self.config.config['runtime-config']['log']
        self.save_model_freq = self.config.config['runtime-config']['save_model_freq']
        self.save_mode_dir = None
        self.train_or_test = None
        self.best_testing_reward = float("-inf")
        self.no_improvement_count = 0

    def train(self):
        # save config
        if self.log:
            now = datetime.now()
            curr_time = now.strftime("%Y%m%d%H%M%S%f")
            dir_lst = []
            self.save_mode_dir = f"logs/workflow_scheduling/{self.RAorSA}{curr_time}"
            dir_lst.append(self.save_mode_dir)
            dir_lst.append(self.save_mode_dir + "/saved_models/")
            dir_lst.append(self.save_mode_dir + "/test_performance/")
            dir_lst.append(self.save_mode_dir + "/train_performance/")
            for _dir in dir_lst:
                os.makedirs(_dir)
            with open(self.save_mode_dir + "/profile.yaml", 'w') as file:
                yaml.dump(self.config.config['yaml-config'], file)
                file.close()

        # init best reward
        if self.config.config['yaml-config']['optim']['maximization']:
            best_reward_so_far = float("-inf")
        else:
            best_reward_so_far = float("inf")

        # init population
        population = self.optim.init_population(self.policy, self.env)

        # run baseline
        indi = None
        envs = self.env_test
        arguments = (indi, deepcopy(envs), self.optim, self.ob_rms_mean, self.ob_rms_std, self.processor_num, self.train_or_test, self.RAPolicy, self.ra_ob_rms_mean,
                     self.ra_ob_rms_std)
        if self.RAorSA == 'RA':
            result = self.worker_func1(arguments)
        else:
            result = self.worker_func2(arguments)
        baseline_reward = result['rewards']
        print(baseline_reward)

        # start training
        g = 1
        while True:
            start_time = time.time()
            # prepare
            self.train_or_test = 'train'
            env = self.dataset.create_train_env()
            arguments = [(
                indi, [deepcopy(env)], self.optim, self.ob_rms_mean, self.ob_rms_std, self.processor_num, self.train_or_test, self.RAPolicy, self.ra_ob_rms_mean,
                self.ra_ob_rms_std) for indi in population]
            end_time_prepare = time.time() - start_time

            # rollout
            start_time_rollout = time.time()
            if self.RAorSA == 'RA':
                if self.processor_num > 1:
                    p = mp.get_context('spawn').Pool(self.processor_num)
                    results = p.map(self.worker_func1, arguments)
                    p.close()
                    p.join()
                else:
                    results = [self.worker_func1(arg) for arg in arguments]
            else:
                if self.processor_num > 1:
                    p = mp.get_context('spawn').Pool(self.processor_num)
                    results = p.map(self.worker_func2, arguments)
                    p.close()
                    p.join()
                else:
                    results = [self.worker_func2(arg) for arg in arguments]
            end_time_rollout = time.time() - start_time_rollout

            results_df = pd.DataFrame(results).sort_values(by=['policy_id'])
            results_to_save = results_df.copy()
            results_to_iterate = results_df.copy()

            # evaluate
            start_time_eval = time.time()
            if (g % self.save_model_freq) == 0:
                from utils.policy_dict import agent_policy
                indi_test = []
                agent_ids_test = self.env.get_agent_ids()
                model_test = self.optim.get_elite_model()
                indi_test.append(agent_policy(agent_ids_test, model_test))
                self.train_or_test = 'test'
                envs = self.env_test
                arguments = [(indi, deepcopy(envs), self.optim, self.ob_rms_mean, self.ob_rms_std, self.processor_num, self.train_or_test, self.RAPolicy,
                              self.ra_ob_rms_mean, self.ra_ob_rms_std) for indi in
                             indi_test]

                start_time_test = time.time()
                if self.RAorSA == 'RA':
                    results = [self.worker_func1(arg) for arg in arguments]
                else:
                    results = [self.worker_func2(arg) for arg in arguments]
                end_time_test = time.time() - start_time_test
                results_df = pd.DataFrame(results)

                testing_reward = results_df['rewards'].tolist()[0]
                testing_makespan = results_df['makespan'].tolist()[0]
                testing_violation = results_df['violation_rate'].tolist()[0]

                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                print(
                    f"episode: {g}, [<<<<----testing---->>>>], testing reward: {testing_reward:.2f}, testing makespan: {testing_makespan:.2f}, testing violation: {testing_violation:.4f}, baseline reward: {baseline_reward:.2f}, current testing_time: {end_time_test:.2f}",
                    flush=True
                )

                if self.log:
                    results_df = results_df.drop(['hist_obs'], axis=1)
                    dir_test = self.save_mode_dir + "/test_performance"
                    if not os.path.exists(dir_test):
                        os.makedirs(dir_test)
                    results_df.to_csv(dir_test + "/testing_record_in_training.csv", index=False, header=False, mode='a')
                if self.best_testing_reward < testing_reward:
                    self.best_testing_reward = testing_reward
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1

            end_time_eval = time.time() - start_time_eval

            # 保存网络模型和观测值均值标准差/更新均值标准差
            result_obs = results_to_save['hist_obs']

            if self.log:
                if self.running_mstd:
                    results_to_save = results_to_save.drop(['hist_obs'], axis=1)
                results_to_save = results_to_save.loc[results_to_save['policy_id'] == -1]

                # 保存训练记录
                dir_train = self.save_mode_dir + "/train_performance"
                if not os.path.exists(dir_train):
                    os.makedirs(dir_train)
                results_to_save.to_csv(dir_train + "/training_record.csv", index=False, header=False, mode='a')

                # 保存模型
                elite = self.optim.get_elite_model()
                if g % self.save_model_freq == 0:
                    save_pth = self.save_mode_dir + "/saved_models" + f"/ep_{g}.pt"
                    torch.save(elite.state_dict(), save_pth)
                    if self.running_mstd:
                        save_pth = self.save_mode_dir + "/saved_models" + f"/ob_rms_{g}.pickle"
                        f = open(save_pth, 'wb')
                        pickle.dump(np.concatenate((self.ob_rms_mean, self.ob_rms_std)), f, protocol=pickle.HIGHEST_PROTOCOL)
                        f.close()

            if self.running_mstd:
                hist_obs = np.concatenate(result_obs, axis=0)
                self.ob_rms.update(hist_obs)
                self.ob_rms_mean = self.ob_rms.mean
                self.ob_rms_std = np.sqrt(self.ob_rms.var)

            # 生成下一代种群，更新最佳奖励
            population, sigma, best_reward, best_makespan, best_overrate, best_individual = self.optim.next_population(results_to_iterate, g)

            maximization = self.config.config['yaml-config']['optim']['maximization']
            if (maximization and best_reward > best_reward_so_far) or (not maximization and best_reward < best_reward_so_far):
                best_reward_so_far = best_reward

            end_time_generation = time.time() - start_time
            print(
                f"\nepisode: {g}, [current_policy_population:], best reward so far: {best_reward_so_far:.2f}, sigma: {sigma:.2f}, current best reward: {best_reward:.2f}, makespan: {best_makespan:.2f} violation rate: {best_overrate:.4f}, time_generation: {end_time_generation:.2f}, prepare_time: {end_time_prepare:.2f}, rollout_time: {end_time_rollout:.2f}, eval_time: {end_time_eval:.2f}",
                flush=True
            )

            g += 1
            if self.no_improvement_count >= 5:
                break

    def eval(self):
        """评估方法，用于评估训练好的策略。"""
        # 加载策略模型
        self.policy.load_state_dict(torch.load(self.config.config['runtime-config']['policy_path']))

        envs = self.env_test

        # 创建带有agent id的个体
        indi = agent_policy(envs[0].get_agent_ids(), self.policy)

        # 加载运行时均值和标准差
        if self.running_mstd:
            with open(self.config.config['runtime-config']['rms_path'], "rb") as f:
                ob_rms = pickle.load(f)
                self.ob_rms_mean = ob_rms[:int(0.5 * len(ob_rms))]
                self.ob_rms_std = ob_rms[int(0.5 * len(ob_rms)):]

        self.policy.eval()  # 设置为评估模式

        # 使用测试集进行评估
        self.train_or_test = 'test'
        arguments = [
            (indi, envs, self.optim, self.ob_rms_mean, self.ob_rms_std, self.processor_num, self.train_or_test, self.RAPolicy, self.ra_ob_rms_mean, self.ra_ob_rms_std)]

        # 开始测试
        start_time_test = time.time()
        if self.RAorSA == 'RA':
            results = [self.worker_func1(arg) for arg in arguments]
        else:
            results = [self.worker_func2(arg) for arg in arguments]
        end_time_test = time.time() - start_time_test

        results_df = pd.DataFrame(results)

        # 打印测试结果
        testing_reward = results_df['rewards'].tolist()[0]
        print(f"current testing reward: {testing_reward:.4f}, testing_time: {end_time_test:.2f}\n", flush=True)

        if self.log:
            results_df = results_df.drop(['hist_obs'], axis=1)
            dir_test = os.path.dirname(self.config.config['runtime-config']['config']) + "/test_performance"
            test_size = self.config.config['yaml-config']['env']['wf_size']
            if not os.path.exists(dir_test):
                os.makedirs(dir_test)
            results_df.to_csv(dir_test + "/testing_record_" + str(test_size) + ".csv", index=False, header=False, mode='a')

    @staticmethod
    def worker_func1(arguments):
        """
        工作函数，用于多进程环境中执行RA策略评估。
        Args:
            arguments: 包含评估所需参数的元组。
        """
        indi, envs, optim, ob_rms_mean, ob_rms_std, processor_num, train_or_test, RAmodel, ra_ob_rms_mean, ra_ob_rms_std = arguments
        if indi:
            agent_id, model = next(iter(indi.items()))
            model = model.to(running_device)

        total_reward = 0
        total_makespan = 0
        total_violation_rate = 0
        obs = None
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
                # for server in env.servers:
                #     if not server.available:
                #         removed_server.append(server.id)
                if env.ready_job_set:
                    select_job = min(env.ready_job_set, key=lambda x: x.workload)
                    if indi:
                        state = get_observe_R(env, select_job)
                        if obs is None:
                            obs = state
                        else:
                            obs = np.append(obs, state, axis=0)
                        # 归一化观测值
                        if ob_rms_mean is not None:
                            state = (state - ob_rms_mean) / ob_rms_std
                        action = model(state, removed=removed_server)
                    else:
                        action = HH(env, select_job, isR=True)
                    if action == len(env.servers):
                        env.addEvent(jobAssignEvent(env.workflows[select_job.workflow_id], select_job, env.devices[select_job.sourceDeviceId], env.current_time))
                    else:
                        env.addEvent(jobAssignEvent(env.workflows[select_job.workflow_id], select_job, env.servers[action], env.current_time))
                    continue

                # 3. 顺序选择
                for device in env.devices:
                    if device.isBusy == False and device.wait_list != []:
                        job = device.wait_list[0]
                        env.addEvent(jobProcessEvent(env.workflows[job.workflow_id], job, device, env.current_time))
                        continue

                for server in env.servers:
                    if server.available and server.isBusy == False and server.wait_list != []:
                        job = server.wait_list[0]
                        env.addEvent(jobProcessEvent(env.workflows[job.workflow_id], job, server, env.current_time))
                        continue

                # 4. 检查任务是否全部完成
                if env.event_list == []:
                    for wf in env.workflows:
                        for job in wf.jobs:
                            if job.state != "已完成":
                                print(f"工作流{wf.id} 任务{job.id}未完成")
                    break

                # 5. 更新时间
                if env.current_time != env.event_list[0].time:
                    env.current_time = env.event_list[0].time

            max_finish_times = []
            for wf in env.workflows:
                max_finish_times.append(max(wj.finishTime for wj in wf.jobs))
            makespan = max(max_finish_times)
            count = 0
            for wf in env.workflows:
                if max(wj.finishTime for wj in wf.jobs) > wf.deadline:
                    count += 1

            total_reward += -makespan * (count + 1)
            total_makespan += makespan
            total_violation_rate += count / len(env.workflows)
            del env
        rewards_mean = total_reward / len(envs)
        makespan_mean = total_makespan / len(envs)
        violation_rate_mean = total_violation_rate / len(envs)

        # 根据环境和优化器类型返回不同结果
        return {'policy_id': indi['0'].policy_id if indi else 0,
                'rewards': rewards_mean,
                'hist_obs': obs,
                'makespan': makespan_mean,
                'violation_rate': violation_rate_mean}

    @staticmethod
    def worker_func2(arguments):
        """
        工作函数，用于多进程环境中执行SA策略评估。
        Args:
            arguments: 包含评估所需参数的元组。
        """
        indi, envs, optim, ob_rms_mean, ob_rms_std, processor_num, train_or_test, RAmodel, ra_ob_rms_mean, ra_ob_rms_std = arguments
        if indi:
            agent_id, model = next(iter(indi.items()))
            RAmodel = RAmodel.to(running_device)
            model = model.to(running_device)

        total_reward = 0
        total_makespan = 0
        total_violation_rate = 0
        obs = None
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
                # for server in env.servers:
                #     if not server.available:
                #         removed_server.append(server.id)
                if env.ready_job_set:
                    select_job = min(env.ready_job_set, key=lambda x: x.workload)
                    state = get_observe_R(env, select_job)
                    # 归一化观测值
                    if ra_ob_rms_mean is not None:
                        state = (state - ra_ob_rms_mean) / ra_ob_rms_std
                    action = RAmodel(state, removed=removed_server)
                    # action = HH(env, select_job, isR=True)
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
                        if indi:
                            state = get_observe_S(env, device)
                            if obs is None:
                                obs = state
                            else:
                                obs = np.append(obs, state, axis=0)
                            # 归一化观测值
                            if ob_rms_mean is not None:
                                state = (state - ob_rms_mean) / ob_rms_std
                            action = model(state, removed=[])
                        else:
                            action = HH(env, device, isR=False)
                        job = device.wait_list[action]
                        env.addEvent(jobProcessEvent(env.workflows[job.workflow_id], job, device, env.current_time))
                        continue

                for server in env.servers:
                    if server.available and server.isBusy == False and len(server.wait_list) == 1:
                        job = server.wait_list[0]
                        env.addEvent(jobProcessEvent(env.workflows[job.workflow_id], job, server, env.current_time))
                        continue
                    elif server.available and server.isBusy == False and len(server.wait_list) > 1:
                        if indi:
                            state = get_observe_S(env, server)
                            if obs is None:
                                obs = state
                            else:
                                obs = np.append(obs, state, axis=0)
                            # 归一化观测值
                            if ob_rms_mean is not None:
                                state = (state - ob_rms_mean) / ob_rms_std
                            action = model(state, removed=[])
                        else:
                            action = HH(env, server, isR=False)
                        job = server.wait_list[action]
                        env.addEvent(jobProcessEvent(env.workflows[job.workflow_id], job, server, env.current_time))
                        continue

                # 4. 检查任务是否全部完成
                if env.event_list == []:
                    for wf in env.workflows:
                        for job in wf.jobs:
                            if job.state != "已完成":
                                print(f"工作流{wf.id} 任务{job.id}未完成")
                    break

                # 5. 更新时间
                if env.current_time != env.event_list[0].time:
                    env.current_time = env.event_list[0].time

            max_finish_times = []
            for wf in env.workflows:
                max_finish_times.append(max(wj.finishTime for wj in wf.jobs))
            makespan = max(max_finish_times)
            count = 0
            for wf in env.workflows:
                if max(wj.finishTime for wj in wf.jobs) > wf.deadline:
                    count += 1

            total_reward += -makespan * (count + 1)
            total_makespan += makespan
            total_violation_rate += count / len(env.workflows)
            del env
        rewards_mean = total_reward / len(envs)
        makespan_mean = total_makespan / len(envs)
        violation_rate_mean = total_violation_rate / len(envs)

        # 根据环境和优化器类型返回不同结果
        return {'policy_id': indi['0'].policy_id if indi else 0,
            'rewards': rewards_mean,
            'hist_obs': obs,
            'makespan': makespan_mean,
            'violation_rate': violation_rate_mean}
