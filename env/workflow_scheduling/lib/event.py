class Event:
    def __init__(self, event_type, workflow, job, server, time, priority):
        """事件基类
        Args:
            event_type (str): 事件类型标识
            workflow (Workflow): 关联的工作流对象
            job (Job): 关联的任务对象
            server (Server): 关联的服务器对象
            time (float): 事件触发时间
            priority (int): 优先级（用于同时间事件的排序）
        """
        self.event_type = event_type  # 事件类型标识
        self.workflow = workflow  # 关联的工作流
        self.job = job  # 关联的任务
        self.server = server  # 关联的服务器
        self.time = time  # 事件触发时间
        self.priority = priority  # 事件优先级

    def __lt__(self, other):
        """定义事件比较逻辑（用于优先队列排序）
        规则：先按时间排序，时间相同则按优先级排序
        """
        if self.time == other.time:
            return self.priority < other.priority
        return self.time < other.time


# ---------------------------- 具体事件子类 ----------------------------

class workflowArriveEvent(Event):
    """工作流到达事件（优先级2）"""

    def __init__(self, workflow, job, server, time):
        super().__init__('workflowArriveEvent', workflow, None, None, time, priority=2)

    def trigger(self, env):
        """触发工作流到达事件"""
        self.workflow.isArrived = True  # 标记工作流已到达

        # 获取当前工作流中所有就绪任务（无父任务或父任务已完成的任务）
        ready_job_set = set(self.workflow.getALlReadyJob())

        # 记录任务就绪时间并更新全局就绪集合
        for job in ready_job_set:
            job.state = "已准备"
            job.readyTime = self.time
            job.readyTime_bac = self.time
        env.ready_job_set.update(ready_job_set)


class jobAssignEvent(Event):
    """任务开始上传事件（优先级1）"""

    def __init__(self, workflow, job, server, time):
        super().__init__('jobAssignEvent', workflow, job, server, time, priority=1)

    def trigger(self, env):
        """触发任务上传事件"""
        self.job.state = "上传中"
        env.ready_job_set.remove(self.job)  # 从就绪集合移除

        # 本地卸载特殊处理（传输时间为0）
        if self.server.server_type == "MobileDevice" and self.workflow.sourceDeviceId == self.server.id:
            self.job.assignedServer = -1  # -1表示本地卸载
            upload_time = 0
        else:
            self.job.assignedServer = self.server.id
            upload_time = self.job.data_up / self.server.bandwidth  # 计算上传时间
            self.server.trans_list.append(self.job)  # 加入服务器传输队列

        # 计算到达时间并创建到达事件
        self.job.arriveTime = self.time + upload_time
        env.addEvent(jobArriveEvent(self.workflow, self.job, self.server, self.job.arriveTime))


class jobArriveEvent(Event):
    """任务到达服务器事件（优先级1）"""

    def __init__(self, workflow, job, server, time):
        super().__init__('jobArriveEvent', workflow, job, server, time, priority=1)

    def trigger(self, env):
        """触发任务到达事件"""
        self.job.state = "排队中"

        # 非本地卸载时从传输队列移除
        if self.server.server_type != "MobileDevice":
            self.server.trans_list.remove(self.job)

        self.server.wait_list.append(self.job)  # 加入服务器等待队列


class jobProcessEvent(Event):
    """任务开始处理事件（优先级1）"""

    def __init__(self, workflow, job, server, time):
        super().__init__('jobProcessEvent', workflow, job, server, time, priority=1)

    def trigger(self, env):
        """触发任务处理事件"""
        self.job.state = "处理中"
        self.server.processing_job = self.job  # 标记服务器当前处理的任务
        self.server.wait_list.remove(self.job)  # 从等待队列移除
        self.server.isBusy = True  # 标记服务器忙碌

        # 计算处理时间和下载时间
        process_time = self.job.workload / self.server.pa
        download_time = 0 if (self.server.server_type == "MobileDevice" and self.workflow.sourceDeviceId == self.server.id) else \
            self.job.data_down / self.server.bandwidth

        # 记录任务时间信息
        self.job.processStartTime = self.time
        self.job.processEndTime = self.time + process_time
        self.job.finishTime = self.time + process_time + download_time

        # 创建下载开始事件
        env.addEvent(jobDownloadEvent(self.workflow, self.job, self.server, self.job.processEndTime))


class jobDownloadEvent(Event):
    """任务开始下载事件（优先级4）"""

    def __init__(self, workflow, job, server, time):
        super().__init__('jobDownloadEvent', workflow, job, server, time, priority=4)

    def trigger(self, env):
        """触发任务下载事件"""
        self.job.state = "下载中"
        self.server.last_free_time = self.time  # 记录服务器最后空闲时间
        self.server.processing_job = None  # 清空当前处理任务
        self.server.isBusy = False  # 标记服务器空闲
        self.server.record.append(self.job)  # 记录完成任务

        # 创建任务完成事件
        env.addEvent(jobFinishEvent(self.workflow, self.job, self.server, self.job.finishTime))


class jobFinishEvent(Event):
    """任务完成事件（优先级2）"""

    def __init__(self, workflow, job, server, time):
        super().__init__('jobFinishEvent', workflow, job, server, time, priority=2)

    def trigger(self, env):
        """触发任务完成事件"""
        self.job.state = "已完成"

        # 获取因当前任务完成而新就绪的任务
        ready_job_set = set(self.workflow.getALlReadyJob())

        # 更新新就绪任务的准备时间
        for job in ready_job_set:
            if job.readyTime is None:
                job.state = "已准备"
                job.readyTime = self.time
                job.readyTime_bac = self.time

        # 更新全局就绪任务集合
        env.ready_job_set.update(ready_job_set)


class serverSuspendEvent(Event):
    def __init__(self, workflow, job, server, time):
        super().__init__('serverSuspendEvent', None, None, server, time, priority=5)

    def trigger(self, env):
        self.server.available = False
        resume_time = next(window[1] for window in self.server.fault_windows if window[0] == self.time)
        fault_type = next(window[2] for window in self.server.fault_windows if window[0] == self.time)
        # 添加服务器恢复事件
        env.addEvent(serverResumeEvent(None, None, self.server, resume_time))

        # 瞬时故障只需延长时间
        if fault_type == 'transient':
            for event in env.event_list:
                job = event.job
                if job:
                    if event.event_type == 'jobArriveEvent' and event.job.assignedServer == self.server.id:
                        job.arriveTime += resume_time - self.time
                        env.addEvent(jobArriveEvent(env.workflows[job.workflow_id], job, self.server, job.arriveTime))
                        env.removeEvent(event)
                    elif event.event_type == 'jobDownloadEvent' and event.job.assignedServer == self.server.id:
                        job.processEndTime += resume_time - self.time
                        job.finishTime += resume_time - self.time
                        env.addEvent(jobDownloadEvent(env.workflows[job.workflow_id], job, self.server, job.processEndTime))
                        env.removeEvent(event)
                    elif event.event_type == 'jobFinishEvent' and event.job.assignedServer == self.server.id:
                        job.finishTime += resume_time - self.time
                        env.addEvent(jobFinishEvent(env.workflows[job.workflow_id], job, self.server, job.finishTime))
                        env.removeEvent(event)

        # 短期故障重新安排任务
        elif fault_type == 'short_term':
            ready_job_set = []
            self.server.processing_job = None
            self.server.isBusy = False
            # 重新设置正在传输或处理的任务
            jobs = [event.job for event in env.event_list if event.job and event.job.assignedServer == self.server.id]
            for job in jobs:
                job.reset()
                ready_job_set.append(job)
                for event in env.event_list:
                    if (event.event_type == 'jobArriveEvent' or event.event_type == 'jobDownloadEvent' or event.event_type == 'jobFinishEvent') and event.job == job:
                        env.removeEvent(event)
            self.server.trans_list = []
            # 重新设置等待队列任务
            for job in self.server.wait_list:
                job.reset()
                ready_job_set.append(job)
            self.server.wait_list = []  # 重置等待队列

            ready_job_set = set(ready_job_set)
            for job in ready_job_set:
                if job.readyTime is None:
                    job.state = "已准备"
                    job.readyTime = job.readyTime_bac  # 恢复就绪时间
            # 更新全局就绪任务集合
            env.ready_job_set.update(ready_job_set)


class serverResumeEvent(Event):
    def __init__(self, workflow, job, server, time):
        super().__init__('serverResumeEvent', None, None, server, time, priority=1)

    def trigger(self, env):
        self.server.available = True
