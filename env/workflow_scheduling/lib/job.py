class Job:
    def __init__(self, job_id, workflow_id, data_up, data_down, workload, sourceDeviceId):
        self.id = job_id
        self.workflow_id = workflow_id
        self.state = "未准备"  # 未准备/已准备/上传中/排队中/处理中/下载中/已完成
        self.data_up = data_up
        self.data_down = data_down
        self.workload = workload
        self.sourceDeviceId = sourceDeviceId

        # 时间属性
        self.readyTime = None
        self.arriveTime = None
        self.processStartTime = None
        self.processEndTime = None
        self.finishTime = None

        self.assignedServer = None
        self.UpwardRank = None
        self.DownwardRank = None
