import copy
import heapq
import gymnasium as gym
from env.lib.event import workflowArriveEvent


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
        self.devices = devices
        self.servers = servers
        self.workflows = workflows

        self.name = "WorkflowScheduling"
        self.observation_space = gym.spaces.Box(low=0, high=10000, shape=(8,))

        # 事件管理
        self.event_list = []
        self.current_time = 0
        self.ready_job_set = set()

        # 构建服务器ID到对象的映射（用于带宽查询）
        self.server_map = {s.id: s for s in servers}
        # 构建设备ID到对象的映射
        self.device_map = {d.id: d for d in devices}
        # 构建 (类型, ID) 到对象的映射
        self.device_by_type_id = {}
        for d in devices:
            self.device_by_type_id[("MobileDevice", d.id)] = d
        for s in servers:
            self.device_by_type_id[(s.server_type, s.id)] = s

        # ====== 新增：网络波动配置 ======
        # None 表示无波动；否则为 dict，例如：
        # {'type': 'sine',        'amplitude': 0.3, 'period': 100.0, 'phase_offset': 0.0}
        # {'type': 'random_walk', 'sigma': 0.05, 'min': 0.3, 'max': 1.5}
        # {'type': 'markov',      'states': [0.5, 1.0, 1.5], 'trans': [[...],[...],[...]]}
        self.network_fluctuation = None

        # 初始化工作流到达事件
        for wf in self.workflows:
            self.addEvent(workflowArriveEvent(wf, None, None, wf.arrivalTime))

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
        """重置仿真环境到初始状态"""
        initial = copy.deepcopy(self.initial_state)
        self.devices = initial['devices']
        self.servers = initial['servers']
        self.workflows = initial['workflows']
        self.event_list = initial['event_list']
        self.current_time = initial['current_time']
        self.ready_job_set = initial['ready_job_set']

        # 重建映射
        self.server_map = {s.id: s for s in self.servers}
        self.device_map = {d.id: d for d in self.devices}
        self.device_by_type_id = {}
        for d in self.devices:
            self.device_by_type_id[("MobileDevice", d.id)] = d
        for s in self.servers:
            self.device_by_type_id[(s.server_type, s.id)] = s

        # ====== 新增：reset 后重新应用网络波动 ======
        if self.network_fluctuation is not None:
            for s in self.servers:
                s.bandwidth_factor = 1.0
            self.update_network_fluctuation()
        # =========================================

    def __str__(self):
        """生成环境状态的字符串表示（用于调试和日志）"""
        device_info = "\n设备信息:\n"
        for device in self.devices:
            device_info += f" {device.name} - 忙碌状态: {device.isBusy}, 等待任务数: {len(device.wait_list)}\n"

        server_info = "\n服务器信息:\n"
        for server in self.servers:
            server_info += f" {server.name} - 忙碌状态: {server.isBusy}, 等待任务数: {len(server.wait_list)}\n"

        workflow_info = "\n工作流信息:\n"
        for workflow in self.workflows:
            workflow_info += f"工作流 {workflow.id} - 到达状态: {workflow.isArrived}, 类型: {workflow.type}, 到达时间: {workflow.arrivalTime}, 来自设备: {workflow.sourceDeviceId}\n"
            for job in workflow.jobs:
                workflow_info += f"  任务 {job.id} - 完成状态: {job.state}\n"

        return f"仿真环境当前时间 {self.current_time}:\n{device_info}{server_info}{workflow_info}"

    def addEvent(self, event):
        """添加新事件到事件堆"""
        heapq.heappush(self.event_list, event)

    def removeEvent(self, event):
        """直接删除特定事件对象"""
        self.event_list = [e for e in self.event_list if e != event]
        heapq.heapify(self.event_list)

    # ============================================================
    #  设备/服务器查询方法
    # ============================================================
    def get_server_by_id(self, server_id):
        """根据ID获取服务器对象"""
        return self.server_map.get(server_id)

    def get_device_by_type_id(self, device_type, device_id):
        """根据 (类型, ID) 获取设备/服务器对象"""
        return self.device_by_type_id.get((device_type, device_id))

    # ============================================================
    #  带宽查询方法
    # ============================================================
    def get_bandwidth_between(self, src_type, src_id, dst_type, dst_id):
        """
        获取两个设备/服务器之间的带宽（bps）

        参数:
            src_type: 源设备类型 ("MobileDevice", "EdgeServer", "CenterServer")
            src_id: 源设备ID
            dst_type: 目标设备类型
            dst_id: 目标设备ID

        返回:
            float: 带宽 (bps)，若为同一设备返回无穷大（传输时间为0）

        带宽映射规则：
            - 同一设备: ∞
            - 用户设备 ↔ 边缘服务器: 边缘服务器的带宽
            - 用户设备 ↔ 云服务器: 云服务器的带宽
            - 边缘 ↔ 边缘: 1024 Mb/s
            - 云 ↔ 云: 根据 region 判断（同档1024，跨一档512，跨两档256）
            - 边缘 ↔ 云: 云服务器的 region 值（128/256/512）
        """
        # 如果是同一个设备，返回无穷大
        if src_type == dst_type and src_id == dst_id:
            return float('inf')

        # 获取源和目标对象
        src = self.device_by_type_id.get((src_type, src_id))
        dst = self.device_by_type_id.get((dst_type, dst_id))

        if src is None or dst is None:
            return 0

        # ============================================================
        #  用户设备 ↔ 服务器
        # ============================================================
        if src_type == "MobileDevice" and dst_type != "MobileDevice":
            # 用户设备 → 服务器：使用目标服务器的带宽
            return dst.bandwidth

        if dst_type == "MobileDevice" and src_type != "MobileDevice":
            # 服务器 → 用户设备：使用源服务器的带宽
            return src.bandwidth

        # 用户设备 ↔ 用户设备（不允许横向传输）
        if src_type == "MobileDevice" and dst_type == "MobileDevice":
            return 0

        # ============================================================
        #  边缘 ↔ 边缘
        # ============================================================
        if src_type == "EdgeServer" and dst_type == "EdgeServer":
            return 1024 * 1024 * 1024  # 1024 Mb/s

        # ============================================================
        #  云 ↔ 云
        # ============================================================
        if src_type == "CenterServer" and dst_type == "CenterServer":
            r1 = src.region
            r2 = dst.region
            if r1 == r2:
                bw_mb = 1024      # 同档
            elif (r1 == 128 and r2 == 256) or (r1 == 256 and r2 == 128) or \
                 (r1 == 256 and r2 == 512) or (r1 == 512 and r2 == 256):
                bw_mb = 512       # 跨一档
            else:
                bw_mb = 256       # 跨两档（128 ↔ 512）
            return bw_mb * 1024 * 1024

        # ============================================================
        #  边缘 ↔ 云
        # ============================================================
        if (src_type == "EdgeServer" and dst_type == "CenterServer") or \
           (src_type == "CenterServer" and dst_type == "EdgeServer"):
            # 取云服务器的 region 值
            cloud = dst if dst_type == "CenterServer" else src
            bw_mb = cloud.region  # 128/256/512
            return bw_mb * 1024 * 1024

        # 默认值
        return 1024 * 1024 * 1024

    # ============================================================
    # 实际带宽（含网络波动）—— 仅供事件仿真实际执行使用
    # ============================================================
    def get_actual_bandwidth_between(self, src_type, src_id, dst_type, dst_id):
        """
        实际带宽 = 名义带宽 × 波动系数
        仅用于 event.py 里的实际传输时间计算；
        observe.py / llh.py / workflow.py 继续使用 get_bandwidth_between()，
        以保证策略的观测仍是静态名义带宽。
        """
        nominal = self.get_bandwidth_between(src_type, src_id, dst_type, dst_id)

        if nominal == float('inf') or nominal <= 0:
            return nominal

        if self.network_fluctuation is None:
            return nominal

        factor = self._get_link_factor(src_type, src_id, dst_type, dst_id)
        return nominal * factor

    def _get_link_factor(self, src_type, src_id, dst_type, dst_id):
        """获取链路波动系数：优先取服务器端的 factor"""
        if src_type == dst_type and src_id == dst_id:
            return 1.0

        src = self.device_by_type_id.get((src_type, src_id))
        dst = self.device_by_type_id.get((dst_type, dst_id))

        if dst is not None and dst_type != "MobileDevice":
            return getattr(dst, 'bandwidth_factor', 1.0)
        if src is not None and src_type != "MobileDevice":
            return getattr(src, 'bandwidth_factor', 1.0)
        return 1.0

    def update_network_fluctuation(self):
        """
        在 current_time 发生变化后调用，更新所有服务器的 bandwidth_factor。
        """
        if self.network_fluctuation is None:
            return

        import random
        import math
        cfg = self.network_fluctuation
        t = self.current_time

        if cfg['type'] == 'sine':
            amp = cfg.get('amplitude', 0.3)
            period = cfg.get('period', 100.0)
            phase_off = cfg.get('phase_offset', 0.0)
            for s in self.servers:
                phase = phase_off * s.id
                s.bandwidth_factor = 1.0 + amp * math.sin(2 * math.pi * t / period + phase)

        elif cfg['type'] == 'random_walk':
            sigma = cfg.get('sigma', 0.05)
            lo = cfg.get('min', 0.3)
            hi = cfg.get('max', 1.5)
            theta = cfg.get('theta', 0.1)  # 回归强度，0=无回归，1=强回归
            for s in self.servers:
                # OU 过程：向 1.0 回归 + 高斯扰动
                drift = theta * (1.0 - s.bandwidth_factor)
                s.bandwidth_factor = min(hi, max(lo, s.bandwidth_factor + drift + random.gauss(0, sigma)))

        elif cfg['type'] == 'markov':
            states = cfg['states']
            trans = cfg['trans']
            for s in self.servers:
                cur = min(range(len(states)), key=lambda i: abs(states[i] - s.bandwidth_factor))
                nxt = random.choices(range(len(states)), weights=trans[cur])[0]
                s.bandwidth_factor = states[nxt]

        else:
            raise ValueError(f"未知的 network_fluctuation type: {cfg['type']}")