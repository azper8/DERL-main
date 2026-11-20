"""
事件驱动仿真模块

事件类型及优先级（数字越小优先级越高）：
    priority=0: jobExecutionFinishEvent  - 任务执行完成（最高优先级，立即释放依赖）
    priority=1: jobAssignEvent / jobArriveEvent / jobProcessEvent  - 调度与数据传输
    priority=2: workflowArriveEvent / jobFinalFinishEvent  - 到达与最终完成
    priority=3: jobOutputDownloadEvent  - 出口任务回传
"""
from collections import defaultdict


class Event:
    """事件基类，所有事件继承此类"""

    def __init__(self, event_type, workflow, job, server, time, priority):
        self.event_type = event_type
        self.workflow = workflow
        self.job = job
        self.server = server
        self.time = time
        self.priority = priority

    def __lt__(self, other):
        if self.time == other.time:
            return self.priority < other.priority
        return self.time < other.time


# ============================================================
#  事件1: 工作流到达
# ============================================================
class workflowArriveEvent(Event):
    def __init__(self, workflow, job, server, time):
        super().__init__('workflowArriveEvent', workflow, None, None, time, priority=2)

    def trigger(self, env):
        self.workflow.isArrived = True
        ready_job_set = set(self.workflow.getALlReadyJob())
        for job in ready_job_set:
            job.state = "已准备"
            job.readyTime = self.time
        env.ready_job_set.update(ready_job_set)


# ============================================================
#  事件2: 任务分配
# ============================================================
class jobAssignEvent(Event):
    def __init__(self, workflow, job, server, time):
        super().__init__('jobAssignEvent', workflow, job, server, time, priority=1)

    def trigger(self, env):
        self.job.state = "传输输入数据中"
        env.ready_job_set.discard(self.job)

        current = (self.server.server_type, self.server.id)
        workflow = self.workflow
        preds = list(workflow.graph.predecessors(self.job.id))

        # ---- 入口任务：输入数据从设备上传 ----
        if len(preds) == 0:
            device_source = ("MobileDevice", self.job.sourceDeviceId)
            if device_source == current:
                max_arrival_time = self.time
            else:
                bandwidth = env.get_actual_bandwidth_between(
                    "MobileDevice", self.job.sourceDeviceId,
                    self.server.server_type, self.server.id
                )
                if bandwidth > 0:
                    trans_time = self.job.input_size / bandwidth
                max_arrival_time = self.time + trans_time
        else:
            # ---- 非入口任务：按源分组累加父任务输出 ----
            source_groups = defaultdict(int)
            for pred_id in preds:
                pred_job = workflow.jobs[pred_id]
                data_source = pred_job.output_location
                if data_source != current:
                    source_groups[data_source] += pred_job.output_size

            max_arrival_time = self.time
            for data_source, total_size in source_groups.items():
                bandwidth = env.get_actual_bandwidth_between(
                    data_source[0], data_source[1],
                    self.server.server_type, self.server.id
                )
                if bandwidth > 0:
                    trans_time = total_size / bandwidth
                    arrival = self.time + trans_time
                    max_arrival_time = max(max_arrival_time, arrival)

        self.job.assignedServer = self.server.id
        self.job.arriveTime = max_arrival_time

        if self.job not in self.server.trans_list:
            self.server.trans_list.append(self.job)

        env.addEvent(jobArriveEvent(self.workflow, self.job, self.server, self.job.arriveTime))


# ============================================================
#  事件3: 输入数据到达
# ============================================================
class jobArriveEvent(Event):
    def __init__(self, workflow, job, server, time):
        super().__init__('jobArriveEvent', workflow, job, server, time, priority=1)

    def trigger(self, env):
        self.job.state = "排队中"

        if self.job in self.server.trans_list:
            self.server.trans_list.remove(self.job)

        self.server.wait_list.append(self.job)


# ============================================================
#  事件4: 开始执行
# ============================================================
class jobProcessEvent(Event):
    def __init__(self, workflow, job, server, time):
        super().__init__('jobProcessEvent', workflow, job, server, time, priority=1)

    def trigger(self, env):
        self.job.state = "执行中"
        self.server.processing_job = self.job
        self.server.wait_list.remove(self.job)
        self.server.isBusy = True

        process_time = self.job.workload / self.server.pa
        self.job.processStartTime = self.time
        self.job.processEndTime = self.time + process_time

        env.addEvent(jobExecutionFinishEvent(self.workflow, self.job, self.server, self.job.processEndTime))


# ============================================================
#  事件5: 执行完成
# ============================================================
class jobExecutionFinishEvent(Event):
    def __init__(self, workflow, job, server, time):
        super().__init__('jobExecutionFinishEvent', workflow, job, server, time, priority=0)

    def trigger(self, env):
        self.job.state = "执行完成"
        self.job.output_location = (self.server.server_type, self.server.id)

        # self.server.record.append(self.job)  # 绘图用

        is_exit = (len(list(self.workflow.graph.successors(self.job.id))) == 0)

        if is_exit:
            env.addEvent(jobOutputDownloadEvent(self.workflow, self.job, self.server, self.time))
        else:
            self.job.finishTime = self.time

        ready_job_set = set(self.workflow.getALlReadyJob())
        for job in ready_job_set:
            job.state = "已准备"
            job.readyTime = self.time
        env.ready_job_set.update(ready_job_set)

        self.server.processing_job = None
        self.server.isBusy = False


# ============================================================
#  事件6: 出口任务回传
# ============================================================
class jobOutputDownloadEvent(Event):
    def __init__(self, workflow, job, server, time):
        super().__init__('jobOutputDownloadEvent', workflow, job, server, time, priority=3)

    def trigger(self, env):
        if self.server.server_type == "MobileDevice":
            self.job.finishTime = self.time
            self.job.state = "已完成"
            return

        device = env.devices[self.job.sourceDeviceId]
        actual_bw = env.get_actual_bandwidth_between(
            self.server.server_type, self.server.id,
            "MobileDevice", self.job.sourceDeviceId
        )
        download_time = self.job.output_size  / actual_bw
        env.addEvent(jobFinalFinishEvent(self.workflow, self.job, self.server, self.time + download_time))


# ============================================================
#  事件7: 出口任务回传完成
# ============================================================
class jobFinalFinishEvent(Event):
    def __init__(self, workflow, job, server, time):
        super().__init__('jobFinalFinishEvent', workflow, job, server, time, priority=2)

    def trigger(self, env):
        self.job.finishTime = self.time
        self.job.state = "已完成"
