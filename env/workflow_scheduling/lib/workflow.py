import networkx as nx
import matplotlib.pyplot as plt


class Workflow:
    """工作流类，表示一个完整的工作流，包含多个有依赖关系的任务"""

    def __init__(self, wf_id, type, arrivalTime, G, jobs, sourceDeviceId):
        self.id = wf_id  # 工作流ID
        self.type = type  # 工作流类型
        self.graph = G  # networkx有向无环图(DAG)
        self.num = self.graph.number_of_nodes()  # 任务数量
        self.jobs = jobs  # 任务列表(索引从0开始)
        self.arrivalTime = arrivalTime  # 工作流到达时间
        self.finishTime = None  # 工作流完成时间
        self.sourceDeviceId = sourceDeviceId  # 源设备ID
        self.isArrived = False  # 是否已到达
        self.deadline = None  # 截至时间

    def isReady(self, job):
        """
        检查任务是否准备就绪(所有前驱任务是否已完成)
        :param job: 要检查的任务
        :return: True如果准备就绪，False否则
        """
        for predecessor in self.graph.predecessors(job.id):
            predecessor_job = self.jobs[predecessor]
            if predecessor_job.state != "已完成":
                return False
        return True

    def getALlReadyJob(self):
        """
        获取所有准备就绪的任务
        :return: 准备就绪的任务列表
        """
        # 收集所有无依赖任务
        ready_job_list = []
        for job in self.jobs:
            if job.state == "未准备" and self.isReady(job):
                ready_job_list.append(job)
        return ready_job_list

    def getCompleteRate(self, avg_processing_speed, avg_transmission_speed):
        """
        计算工作流完成率
        :return: 完成率(0-1之间)
        """
        total_transfer = sum(job.data_up + job.data_down for job in self.jobs)
        total_workload = sum(job.workload for job in self.jobs)
        incomplete_transfer = sum(job.data_up + job.data_down for job in self.jobs if job.state != "已完成")
        incomplete_workload = sum(job.workload for job in self.jobs if job.state != "已完成")
        if total_workload == 0:
            return 0
        complete_rate = 1 - (incomplete_transfer / avg_transmission_speed + incomplete_workload / avg_processing_speed) / (
                total_transfer / avg_transmission_speed + total_workload / avg_processing_speed)
        return complete_rate

    def calculate_upward_ranks(self, avg_processing_speed, avg_transmission_speed):
        """
        计算向上等级排序(从任务到出口的最长路径)
        :param avg_processing_speed: 平均处理速度
        :param avg_transmission_speed: 平均传输速度
        """
        # 初始化所有任务的向上等级
        for job in self.jobs:
            job.UpwardRank = 0

        # 按逆拓扑排序计算
        topological_order = list(nx.topological_sort(self.graph))
        for job_id in reversed(topological_order):
            current_job = self.jobs[job_id]

            # 计算当前任务的处理时间
            upload_time = current_job.data_up / avg_transmission_speed
            download_time = current_job.data_down / avg_transmission_speed
            computation_time = current_job.workload / avg_processing_speed
            current_job_processing_time = upload_time + computation_time + download_time

            # 找出后继任务中的最大向上等级
            max_upward_rank = 0
            for successor_id in self.graph.successors(job_id):
                successor_job = self.jobs[successor_id]
                if successor_job.UpwardRank is not None:
                    max_upward_rank = max(max_upward_rank, successor_job.UpwardRank)

            # 当前任务的向上等级 = 最大后继等级 + 当前任务处理时间
            current_job.UpwardRank = max_upward_rank + current_job_processing_time

    def calculate_downward_ranks(self, avg_processing_speed, avg_transmission_speed):
        """
        计算向下等级排序(从入口到任务的最长路径)
        :param avg_processing_speed: 平均处理速度
        :param avg_transmission_speed: 平均传输速度
        """
        # 初始化所有任务的向下等级
        for job in self.jobs:
            job.DownwardRank = 0

        # 按拓扑排序计算
        topological_order = list(nx.topological_sort(self.graph))
        for job_id in topological_order:
            current_job = self.jobs[job_id]

            # 计算当前任务的处理时间
            upload_time = current_job.data_up / avg_transmission_speed
            download_time = current_job.data_down / avg_transmission_speed
            computation_time = current_job.workload / avg_processing_speed
            current_job_processing_time = upload_time + computation_time + download_time

            # 找出前驱任务中的最大向下等级
            max_downward_rank = 0
            for predecessor_id in self.graph.predecessors(job_id):
                predecessor_job = self.jobs[predecessor_id]
                if predecessor_job.DownwardRank is not None:
                    max_downward_rank = max(max_downward_rank, predecessor_job.DownwardRank)

            # 当前任务的向下等级 = 最大前驱等级 + 当前任务处理时间
            current_job.DownwardRank = max_downward_rank + current_job_processing_time

    def is_on_critical_path(self, job, avg_processing_speed, avg_transmission_speed):
        """
        判断任务是否在关键路径上
        :param job: 要判断的任务
        :param avg_processing_speed: 平均处理速度
        :param avg_transmission_speed: 平均传输速度
        :return: True如果在关键路径上，False否则
        """
        # 使用前需已经计算过upward和downward
        # 计算整个工作流的关键路径长度
        critical_path_length = 0
        for j in self.jobs:
            if j.UpwardRank + j.DownwardRank - (j.workload / avg_processing_speed + (j.data_up + j.data_down) / avg_transmission_speed) > critical_path_length:
                critical_path_length = j.UpwardRank + j.DownwardRank - (j.workload / avg_processing_speed + (j.data_up + j.data_down) / avg_transmission_speed)

        # 判断当前任务是否在关键路径上
        job_total_length = job.UpwardRank + job.DownwardRank - (job.workload / avg_processing_speed + (job.data_up + job.data_down) / avg_transmission_speed)

        # 允许一定的浮点数误差
        return abs(job_total_length - critical_path_length) < 1e-6

    def getEstimatedCompletionTime1(self, avg_processing_speed, avg_transmission_speed):
        """
        估计工作流的完成时间
        :return: 估计的完成时间
        """
        # 按拓扑排序处理任务
        topo_sorted_jobs = list(nx.topological_sort(self.graph))
        finish_times = {job_id: 0 for job_id in topo_sorted_jobs}

        for job_id in topo_sorted_jobs:
            job = self.jobs[job_id]
            # 任务的开始时间取决于前驱任务/没有前驱任务就是工作流达到时间
            ready_time = max(finish_times[pred] for pred in self.graph.predecessors(job_id)) if list(self.graph.predecessors(job_id)) else self.arrivalTime
            upload_time = job.data_up / avg_transmission_speed
            processing_time = job.workload / avg_processing_speed
            download_time = job.data_down / avg_transmission_speed
            finish_times[job_id] = ready_time + upload_time + processing_time + download_time

        # 返回所有任务中最大的完成时间作为工作流完成时间
        EstimatedCompletionTime = max(finish_times.values())
        return EstimatedCompletionTime

    def getEstimatedCompletionTime2(self, env):
        """
        估计工作流的完成时间
        :param env: 环境对象
        :return: 估计的完成时间
        """
        # 计算平均处理速度和带宽
        mean_pa = sum(server.pa for server in env.servers) / len(env.servers)
        mean_bandwidth = sum(server.bandwidth for server in env.servers) / len(env.servers)

        # 按拓扑排序处理任务
        topo_sorted_jobs = list(nx.topological_sort(self.graph))
        finish_times = {job_id: 0 for job_id in topo_sorted_jobs}

        for job_id in topo_sorted_jobs:
            job = self.jobs[job_id]

            if job.finishTime != None:  # 已完成任务直接使用实际完成时间
                finish_times[job_id] = job.finishTime
            else:
                # 根据不同状态估算完成时间
                if job.state == "未准备":
                    # 未准备任务的开始时间取决于前驱任务
                    ready_time = max(finish_times[pred] for pred in self.graph.predecessors(job_id)) if list(self.graph.predecessors(job_id)) else self.arriveTime
                    upload_time = job.data_up / mean_bandwidth
                    processing_time = job.workload / mean_pa
                    download_time = job.data_down / mean_bandwidth
                    finish_times[job_id] = ready_time + upload_time + processing_time + download_time

                elif job.state == "已准备":
                    ready_time = job.readyTime
                    upload_time = job.data_up / mean_bandwidth
                    processing_time = job.workload / mean_pa
                    download_time = job.data_down / mean_bandwidth
                    finish_times[job_id] = ready_time + upload_time + processing_time + download_time

                elif job.state == "上传中":
                    if job.assignedServer == -1:  # 本地处理
                        pa = env.devices[job.sourceDeviceId].pa
                        finish_times[job_id] = job.workload / pa + job.readyTime
                    else:  # 服务器处理
                        bandwidth = env.servers[job.assignedServer].bandwidth
                        pa = env.servers[job.assignedServer].pa
                        finish_times[job_id] = job.arriveTime + job.workload / pa + job.data_down / bandwidth

                elif job.state == "排队中":
                    if job.assignedServer == -1:  # 本地处理
                        pa = env.devices[job.sourceDeviceId].pa
                        if env.devices[job.sourceDeviceId].isBusy == True:
                            finish_times[job_id] = env.devices[job.sourceDeviceId].processing_job.finishTime + job.workload / pa
                        else:
                            finish_times[job_id] = env.current_time + job.workload / pa
                    else:  # 服务器处理
                        pa = env.servers[job.assignedServer].pa
                        bandwidth = env.servers[job.assignedServer].bandwidth
                        if env.servers[job.assignedServer].isBusy == True:
                            finish_times[job_id] = env.servers[
                                                       job.assignedServer].processing_job.finishTime + job.workload / pa + job.data_down / bandwidth
                        else:
                            finish_times[job_id] = env.current_time + job.workload / pa + job.data_down / bandwidth

        # 返回所有任务中最大的完成时间作为工作流完成时间
        EstimatedCompletionTime = max(finish_times.values())
        return EstimatedCompletionTime

    def getEstimatedCompletionTimeAndDelayedJobRate(self, env):
        """
        估计工作流完成时间和延期任务比例
        :param env: 环境对象
        :return: (估计完成时间, 延期任务比例)
        """
        # 计算平均处理速度和带宽
        mean_pa = sum(server.pa for server in env.servers) / len(env.servers)
        mean_bandwidth = sum(server.bandwidth for server in env.servers) / len(env.servers)

        # 按拓扑排序处理任务
        topo_sorted_jobs = list(nx.topological_sort(self.graph))
        finish_times = {job_id: 0 for job_id in topo_sorted_jobs}

        for job_id in topo_sorted_jobs:
            job = self.jobs[job_id]

            if job.finishTime != None:
                finish_times[job_id] = job.finishTime
            else:
                # 根据不同状态估算完成时间(同上)
                if job.state == "未准备":
                    ready_time = max(finish_times[pred] for pred in self.graph.predecessors(job_id)) if list(self.graph.predecessors(job_id)) else self.arriveTime
                    upload_time = job.data_up / mean_bandwidth
                    processing_time = job.workload / mean_pa
                    download_time = job.data_down / mean_bandwidth
                    finish_times[job_id] = ready_time + upload_time + processing_time + download_time

                elif job.state == "已准备":
                    ready_time = job.readyTime
                    upload_time = job.data_up / mean_bandwidth
                    processing_time = job.workload / mean_pa
                    download_time = job.data_down / mean_bandwidth
                    finish_times[job_id] = ready_time + upload_time + processing_time + download_time

                elif job.state == "上传中":
                    if job.assignedServer == -1:
                        pa = env.devices[job.sourceDeviceId].pa
                        finish_times[job_id] = job.workload / pa + job.readyTime
                    else:
                        bandwidth = env.servers[job.assignedServer].bandwidth
                        pa = env.servers[job.assignedServer].pa
                        finish_times[job_id] = job.arriveTime + job.workload / pa + job.data_down / bandwidth

                elif job.state == "排队中":
                    if job.assignedServer == -1:
                        pa = env.devices[job.sourceDeviceId].pa
                        if env.devices[job.sourceDeviceId].isBusy == True:
                            finish_times[job_id] = env.devices[job.sourceDeviceId].processing_job.finishTime + job.workload / pa
                        else:
                            finish_times[job_id] = env.current_time + job.workload / pa
                    else:
                        pa = env.servers[job.assignedServer].pa
                        bandwidth = env.servers[job.assignedServer].bandwidth
                        if env.servers[job.assignedServer].isBusy == True:
                            finish_times[job_id] = env.servers[job.assignedServer].processing_job.finishTime + job.workload / pa + job.data_down / bandwidth
                        else:
                            finish_times[job_id] = env.current_time + job.workload / pa + job.data_down / bandwidth

        # 计算工作流完成时间和延期任务比例
        EstimatedCompletionTime = max(finish_times.values())
        exceed_deadline_count = sum(1 for finish_time in finish_times.values() if finish_time > self.deadline)
        exceed_ratio = exceed_deadline_count / len(finish_times)

        return EstimatedCompletionTime, exceed_ratio

    def draw(self):
        """
        绘制工作流的DAG图
        """
        g = self.graph
        # 为每个节点添加层级信息用于布局
        for layer, nodes in enumerate(nx.topological_generations(g)):
            for node in nodes:
                g.nodes[node]["layer"] = layer
        # 使用多层布局算法
        pos = nx.multipartite_layout(g, subset_key="layer")

        # 绘制图形
        plt.figure(figsize=(10, 6))
        nx.draw(g, pos, with_labels=True, node_size=500, node_color="lightblue")
        plt.title(f"DAG for Workflow {self.id}")
        plt.show()
