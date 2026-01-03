import math
import random
import numpy as np
import networkx as nx
from env.workflow_scheduling.lib.job import Job
from env.workflow_scheduling.lib.server import Server
from env.workflow_scheduling.lib.workflow import Workflow
from env.workflow_scheduling.lib.faultGenerator import FaultGenerator
from env.workflow_scheduling.lib.readXML import generate_workflow_job_from_xml_file
from env.workflow_scheduling.simulator import SimEnv


class CreateDataset():
    def __init__(self, config):
        self.numMobileDevices = config["mobile_num"]
        self.numEdgeServers = config["edge_num"]
        self.numCenterServers = config["center_num"]
        self.numWorkflows = config["wf_num"]
        self.datasetScale = config["wf_size"]
        self.urgency = config.get("urgency", None)
        self.lambdaRate = config.get("lambda_rate", 0.1)
        self.train_num = config.get("evalNum", 1)
        self.test_num = config.get("validNum", 30)
        # 定义不同规模DAG文件的映射字典
        self.switcher = {
            # small规模的工作流
            0: "CyberShake_30.xml",
            1: "Epigenomics_24.xml",
            2: "Ligo_30.xml",
            3: "Montage_25.xml",
            4: "Sipht_29.xml",
            # middle规模的工作流
            5: "CyberShake_50.xml",
            6: "Epigenomics_47.xml",
            7: "Ligo_50.xml",
            8: "Montage_50.xml",
            9: "Sipht_58.xml",
            # large规模的工作流
            10: "CyberShake_100.xml",
            11: "Epigenomics_100.xml",
            12: "Ligo_100.xml",
            13: "Montage_100.xml",
            14: "Sipht_97.xml"
        }

    @staticmethod
    def transmission_rate(d):
        """
        计算传输速率(基于距离d)
        参数:
            d: 设备与服务器之间的距离(米)
        返回:
            传输速率(单位: bps)
        """
        B = 1e8  # 带宽(100MHz)，单位bps
        P = 1  # 传输功率，单位W
        w_0 = 1e-13  # 噪声功率(-100dBm)
        noise_from_other = 1e-11  # 其他干扰噪声(-80dBm)

        g_i_cloud = d ** -4  # 信道增益(路径损耗模型)
        # 计算传输速率(香农公式)
        r_i_cloud = B * math.log2(1 + (P * g_i_cloud) / (w_0 + noise_from_other))

        return r_i_cloud  # 返回传输速率(单位bps)

    def get_dag_type_path(self, dag_type):
        """
        获取DAG文件路径
        参数:
            dag_type: DAG类型索引
        返回:
            对应DAG文件的完整路径
        """
        base_path = "E:/ERL_for_Fog_Computing/env/workflow_scheduling/dax/"  # DAG文件基础路径
        dag_type_path = base_path + self.switcher.get(dag_type, "DAG path error!!!")
        return dag_type_path

    def createWorkflowTimeTable(self):
        """
        创建工作流到达时间表
        参数:
            numMobileDevices: 移动设备数量
            numWorkflows: 每个设备生成的工作流数量
            lambdaRate: 工作流到达率(泊松过程参数)
        返回:
            按时间排序的工作流到达列表[(时间, 设备ID), ...]
        """
        timeTable = []
        for _ in range(self.numMobileDevices):
            workflow_list = [0]  # 第一个工作流到达时间为0
            for _ in range(1, self.numWorkflows):
                # 生成指数分布的时间间隔
                interval = random.expovariate(self.lambdaRate)
                workflow_list.append(workflow_list[-1] + interval)
            timeTable.append(workflow_list)

        # 将所有设备的工作流时间合并并排序
        all_times = [(time, i) for i, sublist in enumerate(timeTable) for time in sublist]  # [(时间, 设备ID), ...]
        sorted_times = sorted(all_times, key=lambda x: x[0])
        top_times = sorted_times[:self.numWorkflows]  # 只保留前numWorkflows个工作流
        return top_times

    def createWorkflows(self, timeTable, servers):
        """
        创建工作流集合
        参数:
            timeTable: 工作流到达时间表
            datasetScale: 数据集规模(small/medium/large)
        返回:
            工作流对象列表
        """
        mean_pa = sum(server.pa for server in servers) / len(servers)
        mean_bandwidth = sum(server.bandwidth for server in servers) / len(servers)
        workflows = []
        wf_id = 0  # 工作流ID计数器
        for releaseTime, deviceId in timeTable:
            # 根据数据集规模随机选择DAG类型
            if self.datasetScale == "small":
                dag_type = random.randint(0, 4)
            elif self.datasetScale == "medium":
                dag_type = random.randint(0, 9)
            elif self.datasetScale == "large":
                dag_type = random.randint(0, 14)

            # 获取DAG文件路径并生成图结构
            dag_path = self.get_dag_type_path(dag_type)
            G = generate_workflow_job_from_xml_file(dag_path)
            numJobs = nx.number_of_nodes(G)  # 获取任务数量

            # 为每个任务生成随机数据量(转换为bits)
            data_up = np.random.uniform(5220, 20680, size=numJobs)  # 上传数据(Mb)
            data_up = [x * 1024 * 1024 for x in data_up]  # 转换为bits
            data_down = np.random.uniform(5220, 20680, size=numJobs)  # 下载数据(Mb)
            data_down = [x * 1024 * 1024 for x in data_down]  # 转换为bits

            # 为每个任务生成随机工作量
            workload = np.random.uniform(5000, 15000, size=numJobs)

            # 创建任务对象列表
            jobs = []
            for job_id in range(0, numJobs):
                jobs.append(Job(job_id, wf_id, data_up[job_id], data_down[job_id], workload[job_id], deviceId))

            # 获取DAG类型名称并创建工作流对象
            dagtype = self.switcher.get(dag_type, "Invalid")
            wf = Workflow(wf_id, dagtype, releaseTime, G, jobs, deviceId)
            base_time = wf.getEstimatedCompletionTime1(mean_pa, mean_bandwidth)

            # if self.urgency == 'loose':
            #     p_ = [0, 0, 1]
            # elif self.urgency == 'medium':
            #     p_ = [0, 1, 0]
            # elif self.urgency == 'tight':
            #     p_ = [1, 0, 0]
            # else:
            #     p_ = None
            #
            # urgency = np.random.choice(['tight', 'medium', 'loose'], p=p_)
            # if urgency == 'tight':
            #     limit_time = base_time * np.random.choice([2, 3], p=[0.8, 0.2])
            # elif urgency == 'medium':
            #     limit_time = base_time * np.random.choice([2, 3], p=[0.5, 0.5])
            # elif urgency == 'loose':
            #     limit_time = base_time * np.random.choice([2, 3], p=[0.2, 0.8])
            # wf.deadline = wf.arrivalTime + limit_time

            wf.deadline = wf.arrivalTime + base_time * np.random.choice([2, 3], p=[0.8, 0.2])
            workflows.append(wf)
            wf_id += 1

        return workflows

    def createMobileDevice(self):
        """
        创建移动设备集合
        参数:
            numMobileDevices: 移动设备数量
        返回:
            移动设备对象列表
        """
        mds = []
        for md_id in range(self.numMobileDevices):
            bandwidth = None

            pa = random.uniform(125, 250)

            name = "MobileDevice_" + str(md_id)
            md = Server(md_id, bandwidth, pa, "MobileDevice", name)
            mds.append(md)
        return mds

    def createServers(self):
        """
        创建服务器集合(边缘服务器+中心服务器)
        参数:
            numEdgeServers: 边缘服务器数量
            numCenterServers: 中心服务器数量
        返回:
            服务器对象列表
        """
        ess = []  # 边缘服务器列表
        for ed_id in range(0, self.numEdgeServers):
            # 边缘服务器带宽: 1024Mbps
            bandwidth = 1024
            bandwidth = bandwidth * 1024 * 1024

            pa = random.uniform(250, 500)

            name = "EdgeServer_" + str(ed_id)
            es = Server(ed_id, bandwidth, pa, "EdgeServer", name)
            ess.append(es)

        css = []  # 中心服务器列表
        for cs_id in range(self.numEdgeServers, self.numEdgeServers + self.numCenterServers):
            # 中心服务器带宽: 20%概率128Mbps, 60%概率256Mbps, 20%概率512Mbps
            bandwidth = np.random.choice([128, 256, 512], p=[0.2, 0.6, 0.2])
            bandwidth = bandwidth * 1024 * 1024

            pa = random.uniform(500, 1000)

            name = "CenterServer_" + str(cs_id - self.numEdgeServers)
            cs = Server(cs_id, bandwidth, pa, "CenterServer", name)
            css.append(cs)

        servers = ess + css  # 合并边缘和中心服务器
        return servers

    @staticmethod
    def setEdgeServerFaultWindows(servers, wfs):
        mean_pa = sum(server.pa for server in servers) / len(servers)
        mean_bandwidth = sum(server.bandwidth for server in servers) / len(servers)
        finished_time = []
        for wf in wfs:
            finished_time.append(wf.arrivalTime + wf.getEstimatedCompletionTime1(mean_pa, mean_bandwidth))
        simulation_time = 3 * max(finished_time)

        fault_fen = FaultGenerator('industrial', simulation_time)
        for server in servers:
            if server.server_type == 'EdgeServer':
                fault_windows = fault_fen.get_fault_windows()
                if fault_windows:
                    # print(fault_windows)
                    # print('已设置故障于服务器:',server.id)
                    server.fault_windows = fault_windows
                    fault_fen.fault_events = []

    def create_random_env(self):
        """
        创建完整的随机数据集(模拟环境)
        参数:
            numMobileDevices: 移动设备数量
            numEdgeServers: 边缘服务器数量
            numCenterServers: 中心服务器数量
            numWorkflows: 工作流数量
            lambdaRate: 工作流到达率
            datasetScale: 数据集规模
        返回:
            模拟环境对象
        """
        # 1. 创建移动设备集合
        self.mds = self.createMobileDevice()
        # 2. 创建服务器集合
        self.servers = self.createServers()
        # 3. 创建工作流到达时间表
        timeTable_wf = self.createWorkflowTimeTable()
        # 4. 创建工作流集合
        wfs = self.createWorkflows(timeTable_wf, self.servers)
        # 5. 设置边缘服务器故障时间窗(调度器不可知)
        # self.setEdgeServerFaultWindows(self.servers, wfs)
        # 6. 创建模拟环境
        envState = SimEnv(self.mds, wfs, self.servers)
        return envState

    def create_train_env(self):
        return self.create_random_env()

    def create_test_envs(self):
        envs = []
        for _ in range(self.test_num):
            envs.append(self.create_random_env())
        return envs

    def create_ls_envs(self):
        envs = []
        lambda_rate_bac = self.lambdaRate
        self.lambdaRate = 0.10
        for _ in range(8):
            envs.append(self.create_random_env())
        self.lambdaRate = 0.15
        for _ in range(8):
            envs.append(self.create_random_env())
        self.lambdaRate = 0.20
        for _ in range(8):
            envs.append(self.create_random_env())
        self.lambdaRate = 0.25
        for _ in range(8):
            envs.append(self.create_random_env())
        self.lambdaRate = lambda_rate_bac
        return envs


if __name__ == '__main__':
    config = {"mobile_num": 3, "edge_num": 20, "center_num": 20, "wf_num": 20, "wf_size": 'small', "lambda_rate": 0.05, "evalNum": 1, "validNum": 30}
    dataset = CreateDataset(config)
    dataset.create_train_env()
