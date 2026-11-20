import os
import random
import heapq
import copy
import numpy as np
from env.lib.job import Job
from env.lib.server import Server
from env.lib.workflow import Workflow
from env.lib.readXML import generate_workflow_from_xml, sample_task_io_by_topology, relabel_to_int, build_edge_ai_pipeline, build_microservice_dag
from env.lib.event import jobAssignEvent, jobProcessEvent
from env.lib.llh import HEFT
from env.simulator import SimEnv


class CreateDataset:
    def __init__(self, config):
        self.numMobileDevices = config["mobile_num"]
        self.numEdgeServers = config["edge_num"]
        self.numCenterServers = config["center_num"]
        self.numWorkflows = config["wf_num"]
        self.datasetScale = config["wf_size"]
        self.probs = config.get("probs", [0.2, 0.8])
        self.lambdaRate = config.get("lambda_rate", 0.1)
        self.train_num = config.get("evalNum", 1)
        self.test_num = config.get("validNum", 30)

        # 不同规模 DAG 文件的映射字典
        self.switcher = {
            # small
            0: "CyberShake_30.xml",
            1: "Epigenomics_24.xml",
            2: "Ligo_30.xml",
            3: "Montage_25.xml",
            4: "Sipht_29.xml",
            # medium
            5: "CyberShake_50.xml",
            6: "Epigenomics_47.xml",
            7: "Ligo_50.xml",
            8: "Montage_50.xml",
            9: "Sipht_58.xml",
            # large
            10: "CyberShake_100.xml",
            11: "Epigenomics_100.xml",
            12: "Ligo_100.xml",
            13: "Montage_100.xml",
            14: "Sipht_97.xml"
        }

        # DAX 文件基础路径
        _CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # env/lib/
        _PROJECT_ROOT = os.path.dirname(os.path.dirname(_CURRENT_DIR))  # DERL/
        self.dax_base_path = os.path.join(_PROJECT_ROOT, "env", "dax") + os.sep

        # 预创建移动设备和服务器（用于 HEFT 计算）
        self.mds = None
        self.servers = None

        # ====== 固定参考环境（用于 deadline 计算）======
        self.reference_servers = self._create_reference_servers(seed=41)
        self.reference_devices = self._create_reference_devices(seed=41)

    def get_dag_type_path(self, dag_type):
        """获取 DAG 文件完整路径"""
        return self.dax_base_path + self.switcher.get(dag_type, "DAG path error!!!")

    # ============================================================
    #  固定参考环境（deadline 计算专用，固定种子随机）
    # ============================================================
    @staticmethod
    def _create_reference_servers(seed):
        """
        创建固定参考服务器集合（用于 deadline 计算）

        配置：10 台边缘服务器 + 10 台云服务器
        - 边缘服务器：带宽 1 Gb/s，pa ~ U(1500, 3000)
        - 云服务器：带宽 ∈ {128, 256, 512} Mb/s（2:6:2），pa ~ U(3000, 6000)

        用固定种子随机，保证可复现，且参考环境异质性接近实际环境。
        """
        rng = np.random.RandomState(seed)

        servers = []

        # ---- 10 台边缘服务器 ----
        for i in range(10):
            pa = rng.uniform(1500, 3000)
            server = Server(
                es_id=i,
                bandwidth=1024 * 1024 * 1024,  # 1 Gb/s
                pa=pa,
                server_type="EdgeServer",
                name=f"RefEdge_{i}"
            )
            server.region = 0
            servers.append(server)

        # ---- 10 台云服务器 ----
        for i in range(10):
            bw_mbps = rng.choice([128, 256, 512], p=[0.2, 0.6, 0.2])
            pa = rng.uniform(3000, 6000)
            server = Server(
                es_id=10 + i,
                bandwidth=bw_mbps * 1024 * 1024,
                pa=pa,
                server_type="CenterServer",
                name=f"RefCenter_{i}"
            )
            server.region = bw_mbps  # 128 / 256 / 512
            servers.append(server)

        return servers

    @staticmethod
    def _create_reference_devices(seed):
        """
        创建固定参考移动设备（用于 deadline 计算）

        配置：1 个设备，pa ~ U(750, 1500)
        用固定种子随机，保证可复现。
        """
        rng = np.random.RandomState(seed)

        pa = rng.uniform(750, 1500)
        device = Server(
            es_id=0,
            bandwidth=None,
            pa=pa,
            server_type="MobileDevice",
            name=f"RefDevice_0"
        )
        return [device]

    # ============================================================
    #  实际环境（仿真专用）
    # ============================================================
    def createWorkflowTimeTable(self):
        """
        创建工作流到达时间表（泊松过程）
        返回: [(到达时间, 设备ID), ...] 按时间排序
        """
        timeTable = []
        for device_id in range(self.numMobileDevices):
            workflow_list = [0]  # 第一个工作流到达时间为0
            for _ in range(1, self.numWorkflows):
                interval = random.expovariate(self.lambdaRate)
                workflow_list.append(workflow_list[-1] + interval)
            timeTable.append(workflow_list)

        # 合并所有设备的时间并排序
        all_times = [(time, i) for i, sublist in enumerate(timeTable) for time in sublist]
        sorted_times = sorted(all_times, key=lambda x: x[0])
        return sorted_times[:self.numWorkflows]

    def createMobileDevice(self):
        """创建移动设备集合"""
        mds = []
        for md_id in range(self.numMobileDevices):
            pa = random.uniform(750, 1500)  # 750-1500 MI/s
            name = "MobileDevice_" + str(md_id)
            md = Server(md_id, None, pa, "MobileDevice", name)
            mds.append(md)
        return mds

    def createServers(self):
        """创建实际服务器集合（边缘 + 云端）"""
        ess = []
        for ed_id in range(self.numEdgeServers):
            bandwidth = 1024 * 1024 * 1024  # 1024 Mb/s → bps
            pa = random.uniform(1500, 3000)  # 1500-3000 MI/s
            name = "EdgeServer_" + str(ed_id)
            es = Server(ed_id, bandwidth, pa, "EdgeServer", name)
            es.region = 0  # 边缘服务器都在本地区域
            ess.append(es)

        css = []
        for cs_id in range(self.numEdgeServers, self.numEdgeServers + self.numCenterServers):
            # 云服务器带宽：128/256/512 Mb/s
            bandwidth_choice = np.random.choice([128, 256, 512], p=[0.2, 0.6, 0.2])
            bandwidth = bandwidth_choice * 1024 * 1024
            pa = random.uniform(3000, 6000)  # 3000-6000 MI/s
            name = "CenterServer_" + str(cs_id - self.numEdgeServers)
            cs = Server(cs_id, bandwidth, pa, "CenterServer", name)
            cs.region = bandwidth_choice  # 128/256/512 代表不同距离层
            css.append(cs)

        servers = ess + css
        return servers

    # ============================================================
    #  HEFT 计算
    # ============================================================
    @staticmethod
    def _run_heft_on_single_workflow(wf, servers, devices):
        """
        使用 HEFT 算法计算单个工作流的 makespan
        注意：传入的 wf 是深拷贝副本，修改不会影响原始数据
        """
        # 由于传入的是深拷贝，直接修改其 ID
        wf.id = 0
        for job in wf.jobs:
            job.workflow_id = 0

        # 创建临时环境
        temp_env = SimEnv(devices, [wf], servers)

        # 手动重置状态
        for server in temp_env.servers:
            server.wait_list = []
            server.trans_list = []
            server.isBusy = False
            server.processing_job = None
        for device in temp_env.devices:
            device.wait_list = []
            device.trans_list = []
            device.isBusy = False
            device.processing_job = None
        temp_env.ready_job_set = set()
        temp_env.event_list = []
        temp_env.current_time = 0

        # 计算平均处理能力和带宽（只用有 bandwidth 的服务器）
        server_list_with_bw = [s for s in servers if s.bandwidth is not None]
        mean_pa = sum(server.pa for server in servers) / len(servers) if servers else 1
        mean_bandwidth = (
            sum(s.bandwidth for s in server_list_with_bw) / len(server_list_with_bw)
            if server_list_with_bw else 1
        )

        wf.calculate_upward_ranks(mean_pa, mean_bandwidth)
        wf.calculate_downward_ranks(mean_pa, mean_bandwidth)

        # 重新初始化工作流到达事件
        from env.lib.event import workflowArriveEvent
        temp_env.addEvent(workflowArriveEvent(wf, None, None, wf.arrivalTime))

        current_time = 0

        while True:
            # 1. 触发当前时间的所有事件
            while temp_env.event_list and temp_env.event_list[0].time == current_time:
                event = heapq.heappop(temp_env.event_list)
                event.trigger(temp_env)

            # 2. 路由选择
            if temp_env.ready_job_set:
                select_job = max(
                    temp_env.ready_job_set,
                    key=lambda x: x.UpwardRank if x.UpwardRank is not None else 0.0
                )

                action = HEFT(temp_env, select_job, isR=True)

                if action == len(temp_env.servers):
                    temp_env.addEvent(jobAssignEvent(
                        temp_env.workflows[select_job.workflow_id],
                        select_job,
                        temp_env.devices[select_job.sourceDeviceId],
                        current_time
                    ))
                else:
                    temp_env.addEvent(jobAssignEvent(
                        temp_env.workflows[select_job.workflow_id],
                        select_job,
                        temp_env.servers[action],
                        current_time
                    ))
                continue

            # 3. 顺序选择（设备）
            for device in temp_env.devices:
                if not device.isBusy and device.wait_list:
                    action = HEFT(temp_env, device, isR=False)
                    job = device.wait_list[action]
                    temp_env.addEvent(jobProcessEvent(
                        temp_env.workflows[job.workflow_id],
                        job, device,
                        current_time
                    ))
                    continue

            # 4. 顺序选择（服务器）
            for server in temp_env.servers:
                if server.available and not server.isBusy and server.wait_list:
                    action = HEFT(temp_env, server, isR=False)
                    job = server.wait_list[action]
                    temp_env.addEvent(jobProcessEvent(
                        temp_env.workflows[job.workflow_id],
                        job, server,
                        current_time
                    ))
                    continue

            # 5. 检查是否完成
            if not temp_env.event_list:
                break

            # 6. 更新时间
            current_time = temp_env.event_list[0].time

        # 获取工作流完成时间
        heft_makespan = wf.getWorkflowFinishTime()

        return heft_makespan

    # ============================================================
    #  创建工作流
    # ============================================================
    def createWorkflows(self, timeTable, servers):
        """
        创建工作流集合

        关键改动：
            - deadline 用固定参考环境算 HEFT makespan
            - 实际环境（servers）只用于仿真，不用于 deadline 计算
        """
        # 确保 mds 已创建
        if self.mds is None:
            self.mds = self.createMobileDevice()

        workflows = []
        wf_id = 0

        for releaseTime, deviceId in timeTable:
            # 选择 DAG 类型
            if self.datasetScale == "small":
                dag_type = random.randint(0, 4)
            elif self.datasetScale == "medium":
                dag_type = random.randint(0, 9)
            elif self.datasetScale == "large":
                dag_type = random.randint(0, 14)
            else:
                dag_type = random.randint(0, 4)

            # 从 XML 解析 DAG
            dag_path = self.get_dag_type_path(dag_type)
            G, runtime_map, task_name_map = generate_workflow_from_xml(dag_path)
            # G = relabel_to_int(build_microservice_dag())

            if G is None:
                continue

            # 任务级 IO 采样
            input_map, output_map = sample_task_io_by_topology(
                G,
                target_min_mb=5220,
                target_max_mb=20680
            )

            numJobs = G.number_of_nodes()

            # 为每个任务生成随机 workload
            workload_list = np.random.uniform(30000, 90000, size=numJobs)

            # 创建 Job 对象列表
            jobs = []
            for task_id in range(numJobs):
                job = Job(
                    job_id=task_id,
                    workflow_id=wf_id,
                    input_size=input_map.get(task_id, 0),
                    output_size=output_map.get(task_id, 0),
                    workload=workload_list[task_id],
                    sourceDeviceId=deviceId
                )
                jobs.append(job)

            # 创建工作流
            dagtype = self.switcher.get(dag_type, "Invalid")
            wf = Workflow(wf_id, dagtype, releaseTime, G, jobs, deviceId)

            # ====== 用固定参考环境算 deadline ======
            wf_for_heft = copy.deepcopy(wf)
            wf_for_heft.id = 0
            for job in wf_for_heft.jobs:
                job.workflow_id = 0

            # 把 sourceDeviceId 映射到 0
            wf_for_heft.sourceDeviceId = 0
            for job in wf_for_heft.jobs:
                job.sourceDeviceId = 0

            # 算 HEFT makespan
            heft_makespan = self._run_heft_on_single_workflow(
                wf_for_heft,
                self.reference_servers,
                self.reference_devices
            )
            wf.deadline = wf.arrivalTime + heft_makespan * np.random.choice([2, 3], p=self.probs)

            workflows.append(wf)
            wf_id += 1

        return workflows

    def create_random_env(self):
        """创建完整的随机环境"""
        self.mds = self.createMobileDevice()
        self.servers = self.createServers()
        timeTable_wf = self.createWorkflowTimeTable()
        wfs = self.createWorkflows(timeTable_wf, self.servers)
        envState = SimEnv(self.mds, wfs, self.servers)
        return envState

    def create_train_env(self):
        """创建训练环境"""
        return self.create_random_env()

    def create_test_envs(self):
        """创建测试环境列表"""
        envs = []
        for _ in range(self.test_num):
            envs.append(self.create_random_env())
        return envs


if __name__ == '__main__':
    seed = 52
    random.seed(seed)
    np.random.seed(seed)
    config = {
        "mobile_num": 3,
        "edge_num": 20,
        "center_num": 20,
        "wf_num": 20,
        "wf_size": 'small',
        "lambda_rate": 0.1,
        "evalNum": 1,
        "validNum": 30,
    }
    dataset = CreateDataset(config)
    env = dataset.create_train_env()
    print(f"创建环境成功，包含 {len(env.workflows)} 个工作流")
    for wf in env.workflows:
        print(f"  工作流 {wf.id}: {wf.num} 个任务, 类型 {wf.type}, deadline: {wf.deadline:.2f}")
