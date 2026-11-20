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
from env.lib.observe import get_observe_R, get_observe_S
from env.lib.event import jobAssignEvent, jobProcessEvent
from env.lib.llh import *

running_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AssembleRL(BaseAssemble):
    """强化学习组装类，继承自BaseAssemble，用于训练和评估强化学习策略。"""

    def __init__(self, config, dataset, policy, policy2, optim, ob_rms_r=None, ob_rms_s=None):
        super(AssembleRL, self).__init__()
        self.config = config
        self.dataset = dataset
        self.env = dataset.create_train_env()
        self.env_test = dataset.create_test_envs()
        self.policy_r = policy
        self.policy_s = policy2
        self.optim = optim

        # 运行时的均值标准差设置,用于归一化观测值
        self.running_mstd = self.config.config['yaml-config']["optim"]['input_running_mean_std']
        if self.running_mstd and not ob_rms_r and not ob_rms_s:
            self.ob_rms_r = RunningMeanStd(shape=self.env.observation_space.shape)
            self.ob_rms_mean_r = self.ob_rms_r.mean
            self.ob_rms_std_r = np.sqrt(self.ob_rms_r.var)
            self.ob_rms_s = RunningMeanStd(shape=self.env.observation_space.shape)
            self.ob_rms_mean_s = self.ob_rms_s.mean
            self.ob_rms_std_s = np.sqrt(self.ob_rms_s.var)
        elif self.running_mstd:
            self.ob_rms_r = ob_rms_r
            self.ob_rms_mean_r = self.ob_rms_r.mean
            self.ob_rms_std_r = np.sqrt(self.ob_rms_r.var)
            self.ob_rms_s = ob_rms_s
            self.ob_rms_mean_s = self.ob_rms_s.mean
            self.ob_rms_std_s = np.sqrt(self.ob_rms_s.var)
        else:
            self.ob_rms_r = None
            self.ob_rms_mean_r = None
            self.ob_rms_std_r = None
            self.ob_rms_s = None
            self.ob_rms_mean_s = None
            self.ob_rms_std_s = None

        self.generation_num = self.config.config['yaml-config']['optim']['generation_num']
        self.population_size = self.config.config['yaml-config']['optim']['population_size']
        self.times = self.config.config['yaml-config']['optim']['times']
        self.beta = self.config.config['yaml-config']['optim']['beta']
        self.choice = self.config.config['yaml-config']['optim']['choice']
        self.processor_num = self.config.config['runtime-config']['processor_num']
        self.log = self.config.config['runtime-config']['log']
        self.save_model_freq = self.config.config['runtime-config']['save_model_freq']
        self.save_mode_dir = None
        self.train_or_test = None
        self.best_testing_reward = float("-inf")
        self.no_improvement_count = 0

    def train(self):
        # 打印GPU分配信息
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"检测到 {num_gpus} 张GPU")
            for i in range(num_gpus):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("未检测到GPU，使用CPU")

        # save config
        if self.log:
            now = datetime.now()
            curr_time = now.strftime("%Y%m%d%H%M%S%f")
            dir_lst = []
            self.save_mode_dir = f"logs/RS{curr_time}"
            dir_lst.append(self.save_mode_dir)
            dir_lst.append(self.save_mode_dir + "/saved_models/")
            dir_lst.append(self.save_mode_dir + "/test_performance/")
            dir_lst.append(self.save_mode_dir + "/train_performance/")
            for _dir in dir_lst:
                os.makedirs(_dir)
            with open(self.save_mode_dir + "/profile.yaml", 'w') as file:
                yaml.dump(self.config.config['yaml-config'], file)
                file.close()
            with open(self.save_mode_dir + "/saved_models/test_env.pkl", 'wb') as f:
                pickle.dump(self.env_test, f)
            with open(self.save_mode_dir + "/saved_models/policy_r.pkl", 'wb') as f:
                pickle.dump(self.policy_r, f)
            with open(self.save_mode_dir + "/saved_models/policy_s.pkl", 'wb') as f:
                pickle.dump(self.policy_s, f)

        # init best reward
        if self.config.config['yaml-config']['optim']['maximization']:
            best_reward_so_far = float("-inf")
        else:
            best_reward_so_far = float("inf")

        # init population
        population = self.optim.init_population((self.policy_r, self.policy_s), self.env)

        # run baseline
        indi = None
        envs = self.env_test
        arguments = (
            indi, deepcopy(envs), self.optim, self.ob_rms_mean_r, self.ob_rms_std_r, self.ob_rms_mean_s, self.ob_rms_std_s, self.processor_num, self.train_or_test,
            self.beta, self.choice)
        result = self.worker_func(arguments)
        baseline_reward = result['rewards']
        print(baseline_reward)

        # start training
        g = 1
        while True:
            if g > self.generation_num:
                break
            start_time = time.time()
            # prepare
            self.train_or_test = 'train'
            envs = []
            for _ in range(self.times):
                env = self.dataset.create_train_env()
                envs.append(env)
            arguments = [
                (indi, deepcopy(envs), self.optim, self.ob_rms_mean_r, self.ob_rms_std_r, self.ob_rms_mean_s, self.ob_rms_std_s, self.processor_num, self.train_or_test,
                 self.beta, self.choice)
                for indi in population]
            end_time_prepare = time.time() - start_time

            # rollout
            start_time_rollout = time.time()
            if self.processor_num > 1:
                p = mp.get_context('spawn').Pool(self.processor_num)
                results = p.map(self.worker_func, arguments)
                p.close()
                p.join()
            else:
                results = [self.worker_func(arg) for arg in arguments]
            end_time_rollout = time.time() - start_time_rollout

            results_df = pd.DataFrame(results).sort_values(by=['policy_id'])
            results_to_save = results_df.copy()
            results_to_iterate = results_df.copy()

            # evaluate
            start_time_eval = time.time()
            if (g % self.save_model_freq) == 0:
                indi_test = []
                model_test = self.optim.get_elite_model()
                indi_test.append(model_test)
                self.train_or_test = 'test'
                envs = self.env_test
                arguments = [(indi, deepcopy(envs), self.optim, self.ob_rms_mean_r, self.ob_rms_std_r, self.ob_rms_mean_s, self.ob_rms_std_s, self.processor_num,
                              self.train_or_test, self.beta, self.choice) for indi in indi_test]

                start_time_test = time.time()
                results = [self.worker_func(arg) for arg in arguments]
                end_time_test = time.time() - start_time_test
                results_df = pd.DataFrame(results)

                testing_reward = results_df['rewards'].tolist()[0]
                testing_makespan = results_df['makespan'].tolist()[0]
                testing_violation = results_df['violation_rate'].tolist()[0]

                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                print(
                    f"episode: {g}, [<<<<----testing---->>>>], testing reward: {testing_reward:.2f},testing makespan: {testing_makespan:.2f}, testing violation: {testing_violation:.4f}, baseline reward: {baseline_reward:.2f}, testing_time: {end_time_test:.2f}",
                    flush=True)

                if self.log:
                    results_df = results_df.drop(['hist_obs_r'], axis=1)
                    results_df = results_df.drop(['hist_obs_s'], axis=1)
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
            result_obs_r = results_to_save['hist_obs_r'].copy()
            result_obs_s = results_to_save['hist_obs_s'].copy()

            if self.log:
                if self.running_mstd:
                    results_to_save = results_to_save.drop(['hist_obs_r'], axis=1)
                    results_to_save = results_to_save.drop(['hist_obs_s'], axis=1)
                results_to_save = results_to_save.loc[results_to_save['policy_id'] == -1]

                # 保存训练记录
                dir_train = self.save_mode_dir + "/train_performance"
                if not os.path.exists(dir_train):
                    os.makedirs(dir_train)
                results_to_save.to_csv(dir_train + "/training_record.csv", index=False, header=False, mode='a')

                # 保存模型
                elite_r, elite_s = self.optim.get_elite_model()
                if g % self.save_model_freq == 0:
                    save_pth_r = self.save_mode_dir + "/saved_models" + f"/ep_{g}_r.pt"
                    save_pth_s = self.save_mode_dir + "/saved_models" + f"/ep_{g}_s.pt"
                    torch.save(elite_r.state_dict(), save_pth_r)
                    torch.save(elite_s.state_dict(), save_pth_s)
                    if self.running_mstd:
                        save_pth_r = self.save_mode_dir + "/saved_models" + f"/ob_rms_{g}_r.pickle"
                        save_pth_s = self.save_mode_dir + "/saved_models" + f"/ob_rms_{g}_s.pickle"
                        f = open(save_pth_r, 'wb')
                        pickle.dump(np.concatenate((self.ob_rms_mean_r, self.ob_rms_std_r)), f, protocol=pickle.HIGHEST_PROTOCOL)
                        f.close()
                        f = open(save_pth_s, 'wb')
                        pickle.dump(np.concatenate((self.ob_rms_mean_s, self.ob_rms_std_s)), f, protocol=pickle.HIGHEST_PROTOCOL)
                        f.close()

            if self.running_mstd:
                hist_obs_r = np.concatenate(result_obs_r, axis=0)
                self.ob_rms_r.update(hist_obs_r)
                self.ob_rms_mean_r = self.ob_rms_r.mean
                self.ob_rms_std_r = np.sqrt(self.ob_rms_r.var)
                hist_obs_s = np.concatenate(result_obs_s, axis=0)
                self.ob_rms_s.update(hist_obs_s)
                self.ob_rms_mean_s = self.ob_rms_s.mean
                self.ob_rms_std_s = np.sqrt(self.ob_rms_s.var)

            # 生成下一代种群，更新最佳奖励
            population, sigma, best_reward, best_makespan, best_overrate, best_individual = self.optim.next_population(results_to_iterate, g)

            maximization = self.config.config['yaml-config']['optim']['maximization']
            if (maximization and best_reward > best_reward_so_far) or (not maximization and best_reward < best_reward_so_far):
                best_reward_so_far = best_reward

            end_time_generation = time.time() - start_time
            print(
                f"\nepisode: {g}, [current_policy_population:], best_reward_so_far: {best_reward_so_far:.2f}, sigma: {sigma:.2f}, current best reward: {best_reward:.2f}, makespan: {best_makespan:.2f}, violation rate: {best_overrate:.4f}, time generation: {end_time_generation:.2f}, time prepare: {end_time_prepare:.2f}, time rollout: {end_time_rollout:.2f}, time test: {end_time_eval:.2f}",
                flush=True)

            g += 1
            # if self.no_improvement_count >= 5:
            #     break

    def eval(self):
        pass

    @staticmethod
    def worker_func(arguments):
        indi, envs, optim, ob_rms_mean_r, ob_rms_std_r, ob_rms_mean_s, ob_rms_std_s, processor_num, train_or_test, beta, choice = arguments
        if indi:
            (model_r, model_s) = indi
            # 关键：重新将模型移到GPU
            # 因为经过pickle序列化后，模型参数在CPU上
            if torch.cuda.is_available():
                # 使用模型存储的device_id
                if hasattr(model_r, 'device_id'):
                    device_r = torch.device(f"cuda:{model_r.device_id % torch.cuda.device_count()}")
                else:
                    device_r = torch.device("cuda:0")
                if hasattr(model_s, 'device_id'):
                    device_s = torch.device(f"cuda:{model_s.device_id % torch.cuda.device_count()}")
                else:
                    device_s = torch.device("cuda:0")

                model_r = model_r.to(device_r)
                model_s = model_s.to(device_s)
                # 更新模型内部的device引用
                model_r.device = device_r
                model_s.device = device_s

        total_reward = 0
        total_makespan = 0
        total_violation_rate = 0
        obs_r = None
        obs_s = None

        for env in envs:
            env.reset()
            mean_pa = sum(server.pa for server in env.servers) / len(env.servers)
            mean_bandwidth = sum(server.bandwidth for server in env.servers) / len(env.servers)
            for wf in env.workflows:
                wf.calculate_downward_ranks(mean_pa, mean_bandwidth)
                wf.calculate_upward_ranks(mean_pa, mean_bandwidth)

            while True:
                # 1. 完成当前时刻所有任务
                while env.event_list and env.event_list[0].time == env.current_time:
                    event = heapq.heappop(env.event_list)
                    event.trigger(env)

                # 2. 路由选择
                removed_server = []
                if env.ready_job_set:
                    select_job = min(env.ready_job_set, key=lambda x: x.workload)
                    if indi:
                        state = get_observe_R(env, select_job)
                        if obs_r is None:
                            obs_r = state
                        else:
                            obs_r = np.append(obs_r, state, axis=0)
                        # 归一化观测值
                        if ob_rms_mean_r is not None:
                            state = (state - ob_rms_mean_r) / ob_rms_std_r
                        action = model_r(state, removed=removed_server)
                    else:
                        action = HH(env, select_job, isR=True)
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
                            if obs_s is None:
                                obs_s = state
                            else:
                                obs_s = np.append(obs_s, state, axis=0)
                            # 归一化观测值
                            if ob_rms_mean_s is not None:
                                state = (state - ob_rms_mean_s) / ob_rms_std_s
                            action = model_s(state, removed=[])
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
                            if obs_s is None:
                                obs_s = state
                            else:
                                obs_s = np.append(obs_s, state, axis=0)
                            # 归一化观测值
                            if ob_rms_mean_s is not None:
                                state = (state - ob_rms_mean_s) / ob_rms_std_s
                            action = model_s(state, removed=[])
                        else:
                            action = HH(env, server, isR=False)
                        job = server.wait_list[action]
                        env.addEvent(jobProcessEvent(env.workflows[job.workflow_id], job, server, env.current_time))
                        continue

                # 4. 检查任务是否全部完成
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

            # makespan 用所有任务的 finishTime 最大值（出口任务 finishTime 包含回传时间）
            makespan = 0
            for wf in env.workflows:
                wf_finish = wf.getWorkflowFinishTime()
                if wf_finish > makespan:
                    makespan = wf_finish

            count = 0
            for wf in env.workflows:
                if wf.getWorkflowFinishTime() > wf.deadline:
                    count += 1

            if choice=='mul':
                total_reward += -makespan * (beta * count + 1)
            elif choice=='exp':
                total_reward += -makespan * np.exp(beta * count)
            elif choice=='add':
                total_reward += -makespan - beta * count
            total_makespan += makespan
            total_violation_rate += count / len(env.workflows)
            del env

        rewards_mean = total_reward / len(envs)
        makespan_mean = total_makespan / len(envs)
        violation_rate_mean = total_violation_rate / len(envs)

        return {'policy_id': indi[0].policy_id if indi else 0,
                'rewards': rewards_mean,
                'hist_obs_r': obs_r,
                'hist_obs_s': obs_s,
                'makespan': makespan_mean,
                'violation_rate': violation_rate_mean}
