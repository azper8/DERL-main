class Job:
    def __init__(self, job_id, workflow_id, input_size, output_size, workload, sourceDeviceId):
        self.id = job_id
        self.workflow_id = workflow_id

        # ====== 传输 ======
        self.input_size = input_size
        self.output_size = output_size

        # ====== 运行时状态 ======
        self.workload = workload
        self.sourceDeviceId = sourceDeviceId
        self.state = "未准备"  # 未准备/已准备/传输输入数据中/排队中/执行中/执行完成/已完成

        # ====== 时间属性 ======
        self.readyTime = None  # 就绪时间：所有父任务 processEndTime 的 max
        self.arriveTime = None  # 数据到达时间：readyTime + 传输时间
        self.processStartTime = None
        self.processEndTime = None
        self.finishTime = None  # 非出口 = processEndTime；出口 = 回传完成时间

        # ====== 调度属性 ======
        self.assignedServer = None
        self.output_location = None  # (server_type, server_id)

        # ====== Rank值 ======
        self.UpwardRank = None
        self.DownwardRank = None

        # ====== RL属性(DEL-EA使用) ======
        self.S = None  # 状态
        self.A = None  # 动作
        self.R = None  # 奖励
        self.next_S = None  # 下一状态
        self.terminal = False  # 是否终止

    def reset(self):
        """重置任务状态（用于故障恢复）"""
        self.state = "未准备"
        self.arriveTime = None
        self.processStartTime = None
        self.processEndTime = None
        self.finishTime = None
        self.assignedServer = None
        self.output_location = None