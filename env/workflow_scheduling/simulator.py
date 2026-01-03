import copy
import heapq
import gymnasium as gym
from env.workflow_scheduling.lib.event import workflowArriveEvent, serverSuspendEvent


class SimEnv:
    def __init__(self, devices, workflows, servers):
        """
        仿真环境类，用于管理工作流调度仿真

        参数:
            devices (list): 移动设备列表
            workflows (list): 工作流列表
            servers (list): 服务器列表（边缘+中心）
        """
        # 基础组件
        self.devices = devices  # 所有移动设备对象列表
        self.servers = servers  # 所有服务器对象列表
        self.workflows = workflows  # 所有工作流对象列表

        self.name = "WorkflowScheduling"
        self.observation_space = gym.spaces.Box(low=0, high=10000, shape=(8,))

        # 事件管理
        self.event_list = []  # 事件堆（优先队列），存储待处理事件对象
        self.current_time = 0  # 当前仿真时钟时间

        # 任务状态跟踪
        self.ready_job_set = set()  # 就绪任务集合，元素为元组 (workflow_id, job_id, 到达时间)

        # 初始化工作流到达事件
        for wf in self.workflows:
            self.addEvent(workflowArriveEvent(wf, None, None, wf.arrivalTime))

        # 初始化服务器中止事件
        # for server in self.servers:
        #     if server.server_type == 'EdgeServer' and server.fault_windows:
        #         for window in server.fault_windows:
        #             self.addEvent(serverSuspendEvent(None, None, server, window[0]))

        # 保存初始状态（用于reset）
        self.initial_state = copy.deepcopy({
            'devices': self.devices,
            'servers': self.servers,
            'workflows': self.workflows,
            'event_list': self.event_list,
            'current_time': self.current_time,
            'ready_job_set': self.ready_job_set
        })

    def reset(self):
        """
        重置仿真环境到初始状态
        """
        initial = copy.deepcopy(self.initial_state)
        self.devices = initial['devices']
        self.servers = initial['servers']
        self.workflows = initial['workflows']
        self.event_list = initial['event_list']
        self.current_time = initial['current_time']
        self.ready_job_set = initial['ready_job_set']

    def __str__(self):
        """
        生成环境状态的字符串表示（用于调试和日志）
        """
        # 设备状态信息
        device_info = "\n设备信息:\n"
        for device in self.devices:
            device_info += f" {device.name} - 忙碌状态: {device.isBusy}, 等待任务数: {len(device.wait_list)}\n"

        # 服务器状态信息
        server_info = "\n服务器信息:\n"
        for server in self.servers:
            server_info += f" {server.name} - 忙碌状态: {server.isBusy}, 等待任务数: {len(server.wait_list)}\n"

        # 工作流详细信息
        workflow_info = "\n工作流信息:\n"
        for workflow in self.workflows:
            workflow_info += f"工作流 {workflow.id} - 到达状态: {workflow.isArrived}, 类型: {workflow.type}, 到达时间: {workflow.arrivalTime}, 来自设备: {workflow.sourceDeviceId}\n"
            for job in workflow.jobs:
                workflow_info += f"  任务 {job.id} - 完成状态: {job.state}, 上行数据量: {job.data_up}, 下行数据量: {job.data_down}, 工作量: {job.workload}\n"

        return f"仿真环境当前时间 {self.current_time}:\n{device_info}{server_info}{workflow_info}"

    def addEvent(self, event):
        """
        添加新事件到事件堆（按时间优先级排序）
        参数:
            event (Event): 事件对象，必须实现__lt__比较方法
        """
        heapq.heappush(self.event_list, event)  # 使用堆结构保证事件按时间顺序处理

    def removeEvent(self, event):
        """直接删除特定事件对象"""
        events = [e for e in self.event_list if e != event]
        self.event_list = events
        heapq.heapify(self.event_list)

    @staticmethod
    def get_agent_ids():
        """
        获取环境中的智能体ID列表

        说明:
        - 当前为单智能体环境
        - 固定返回ID为"0"的智能体
        - 满足多智能体环境接口要求

        返回:
            list: 包含智能体ID的列表
        """
        return ["0"]
