"""
工作流模块

工作流由多个有依赖关系的任务组成，用 DAG（有向无环图）表示。
任务之间的依赖关系：父任务执行完成后，子任务即可变为"已准备"状态。

核心约定：
    - 入口任务：input_size 独立采样，表示从用户设备上传的原始数据量
    - 非入口任务：input_size 仅作为状态特征，不参与传输时间计算
    - 传输时间：在 jobAssignEvent 里按父任务 output_size 分组累加，异源取 max
    - Rank 计算：只有入口任务加 input_size 的上传时间，非入口任务不加
"""

import networkx as nx
from collections import defaultdict


class Workflow:
    def __init__(self, wf_id, wf_type, arrivalTime, graph, jobs, sourceDeviceId):
        self.id = wf_id
        self.type = wf_type
        self.graph = graph
        self.num = graph.number_of_nodes()
        self.jobs = jobs
        self.arrivalTime = arrivalTime
        self.finishTime = None
        self.sourceDeviceId = sourceDeviceId
        self.isArrived = False
        self.deadline = None

    # ============================================================
    #  依赖释放
    # ============================================================
    def isReady(self, job):
        """
        任务是否满足执行条件：所有父任务执行完成即可。
        注意：非出口任务执行结束即完成（finishTime = processEndTime），
        不需要等数据回传。
        """
        for pred_id in self.graph.predecessors(job.id):
            pred_job = self.jobs[pred_id]
            if pred_job.state not in ["执行完成", "已完成"]:
                return False
        return True

    def getALlReadyJob(self):
        """获取当前工作流中所有已满足执行条件的任务（未准备状态 + 父任务已完成）"""
        ready_list = []
        for job in self.jobs:
            if job.state == "未准备" and self.isReady(job):
                ready_list.append(job)
        return ready_list

    # ============================================================
    #  工作流完成时间 / 完成率
    # ============================================================
    def getWorkflowFinishTime(self):
        """工作流完成时间 = 所有任务 finishTime 的最大值"""
        finish_times = [j.finishTime for j in self.jobs if j.finishTime is not None]
        return max(finish_times) if finish_times else 0

    def getCompleteRate(self):
        """工作流完成率 = 已完成任务的执行工作量 / 总执行工作量"""
        total_workload = sum(job.workload for job in self.jobs)
        if total_workload == 0:
            return 0
        completed_workload = sum(
            job.workload
            for job in self.jobs
            if job.state in ["执行完成", "已完成"]
        )
        return completed_workload / total_workload

    # ============================================================
    #  辅助：任务自身耗时
    # ============================================================
    def _get_job_self_time(self, job, avg_pa, avg_bw):
        """
        任务自身耗时（不含边上的传输时间）：
            - 执行时间：workload / avg_pa
            - 入口任务：额外加 input_size / avg_bw（从用户设备上传）
            - 出口任务：额外加 output_size / avg_bw（回传用户设备）
        非入口任务的 input_size 不参与，因为其输入来自父任务输出，
        传输时间已经在边的 trans_time 里体现。
        """
        self_time = job.workload / avg_pa

        is_entry = (len(list(self.graph.predecessors(job.id))) == 0)
        is_exit = (len(list(self.graph.successors(job.id))) == 0)

        if is_entry and avg_bw > 0:
            self_time += job.input_size / avg_bw
        if is_exit and avg_bw > 0:
            self_time += job.output_size / avg_bw

        return self_time

    # ============================================================
    #  HEFT 相关 Rank 计算
    # ============================================================
    def calculate_upward_ranks(self, avg_pa, avg_bw):
        """
        UpwardRank(job) = 任务自身耗时 + max(边传输时间 + 后继任务 UpwardRank)
        边传输时间 = job.output_size / avg_bw
        """
        for job in self.jobs:
            job.UpwardRank = 0

        for job_id in reversed(list(nx.topological_sort(self.graph))):
            job = self.jobs[job_id]

            self_time = self._get_job_self_time(job, avg_pa, avg_bw)

            max_succ = 0
            for succ_id in self.graph.successors(job_id):
                succ_job = self.jobs[succ_id]
                trans_time = job.output_size / avg_bw if avg_bw > 0 else 0
                max_succ = max(max_succ, succ_job.UpwardRank + trans_time)

            job.UpwardRank = self_time + max_succ

    def calculate_downward_ranks(self, avg_pa, avg_bw):
        """
        DownwardRank(job) = max(父任务 DownwardRank + 边传输时间) + 任务自身耗时
        边传输时间 = pred_job.output_size / avg_bw
        """
        for job in self.jobs:
            job.DownwardRank = 0

        for job_id in nx.topological_sort(self.graph):
            job = self.jobs[job_id]

            max_pred = 0
            for pred_id in self.graph.predecessors(job_id):
                pred_job = self.jobs[pred_id]
                trans_time = pred_job.output_size / avg_bw if avg_bw > 0 else 0
                max_pred = max(max_pred, pred_job.DownwardRank + trans_time)

            self_time = self._get_job_self_time(job, avg_pa, avg_bw)

            job.DownwardRank = max_pred + self_time

    def is_on_critical_path(self, job, avg_pa, avg_bw):
        """
        判断任务是否在关键路径上：
            UpwardRank + DownwardRank - 自身耗时 == 全局关键路径长度
        """
        if job.UpwardRank is None or job.DownwardRank is None:
            return False

        critical_path_length = max(
            j.UpwardRank for j in self.jobs if j.UpwardRank is not None
        )

        self_time = self._get_job_self_time(job, avg_pa, avg_bw)
        job_path_len = job.UpwardRank + job.DownwardRank - self_time

        return abs(job_path_len - critical_path_length) < 1e-6

    # ============================================================
    #  工作流完成时间和延期任务比例估算（用于 Niching GP 对比算法）
    # ============================================================
    def getEstimatedCompletionTimeAndDelayedJobRate(self, env):
        """
        估计工作流完成时间和延期任务比例。

        传输模型：
            ready_time = 所有父任务的最晚完成时刻
            max_trans_time = max(各源 总大小/带宽)   # 同源累加、异源取 max
            arrival = ready_time + max_trans_time
        """
        mean_pa = sum(server.pa for server in env.servers) / len(env.servers)
        mean_bw = sum(server.bandwidth for server in env.servers) / len(env.servers)

        topo_sorted_jobs = list(nx.topological_sort(self.graph))
        finish_times = {job_id: 0 for job_id in topo_sorted_jobs}

        for job_id in topo_sorted_jobs:
            job = self.jobs[job_id]
            is_exit = (len(list(self.graph.successors(job_id))) == 0)

            # 已完成任务直接取实际完成时间
            if job.finishTime is not None:
                finish_times[job_id] = job.finishTime
                continue

            # ---- 确定目标位置 ----
            job_is_assigned = (job.assignedServer is not None and job.assignedServer != -1)

            if job_is_assigned:
                server = env.server_map.get(job.assignedServer)
                if server:
                    pa = server.pa
                    bandwidth = server.bandwidth
                    dst_type, dst_id = server.server_type, server.id
                else:
                    pa = mean_pa
                    bandwidth = mean_bw
                    dst_type, dst_id = "MobileDevice", job.sourceDeviceId
            else:
                pa = mean_pa
                bandwidth = mean_bw
                dst_type, dst_id = "MobileDevice", job.sourceDeviceId

            # ---- 计算就绪时间 ----
            preds = list(self.graph.predecessors(job_id))

            if preds:
                # 按源分组累加父任务输出
                source_groups = defaultdict(int)
                latest_finish = 0

                for pred_id in preds:
                    pred_job = self.jobs[pred_id]

                    if pred_job.assignedServer is not None and pred_job.assignedServer != -1 \
                            and pred_job.output_location is not None:
                        data_source = pred_job.output_location
                    else:
                        data_source = ("MobileDevice", pred_job.sourceDeviceId)

                    source_groups[data_source] += pred_job.output_size

                    pred_finish = (
                        pred_job.finishTime
                        if pred_job.finishTime is not None
                        else pred_job.processEndTime
                    )
                    if pred_finish is not None:
                        latest_finish = max(latest_finish, pred_finish)

                # 异源取 max
                max_trans_time = 0
                for data_source, total_size in source_groups.items():
                    if (dst_type, dst_id) == data_source:
                        continue
                    bw = env.get_bandwidth_between(
                        data_source[0], data_source[1],
                        dst_type, dst_id
                    )
                    if bw <= 0 or bw == float('inf'):
                        bw = mean_bw
                    trans_time = total_size / bw if bw > 0 else 0
                    max_trans_time = max(max_trans_time, trans_time)

                ready_time = latest_finish + max_trans_time
            else:
                # 入口任务：从用户设备上传
                if job_is_assigned:
                    bw = bandwidth
                else:
                    bw = mean_bw
                upload_time = job.input_size / bw if bw > 0 else 0
                ready_time = self.arrivalTime + upload_time

            # ---- 执行时间 ----
            processing_time = job.workload / pa if pa > 0 else 0

            # ---- 出口任务回传 ----
            if is_exit:
                if job_is_assigned:
                    bw = bandwidth
                else:
                    bw = mean_bw
                download_time = job.output_size / bw if bw > 0 else 0
            else:
                download_time = 0

            finish_times[job_id] = ready_time + processing_time + download_time

        EstimatedCompletionTime = max(finish_times.values()) if finish_times else 0

        # ---- 统计延期任务比例 ----
        exceed_deadline_count = 0
        total_jobs = len(finish_times)
        if total_jobs > 0 and self.deadline is not None:
            for job_id, finish_time in finish_times.items():
                if finish_time > self.deadline:
                    exceed_deadline_count += 1

        DelayedJobRate = exceed_deadline_count / total_jobs if total_jobs > 0 else 0

        return EstimatedCompletionTime, DelayedJobRate