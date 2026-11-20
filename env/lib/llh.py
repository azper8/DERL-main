from collections import defaultdict


def HH(env, instance, isR):
    """HH 启发式（简化版，只考虑最早完成时间，不考虑任务优先级插入）"""
    if isR:
        select_job = instance
        workflow = env.workflows[select_job.workflow_id]
        preds = list(workflow.graph.predecessors(select_job.id))
        is_entry = (len(preds) == 0)
        is_exit = (len(list(workflow.graph.successors(select_job.id))) == 0)

        # ---- 非入口任务：按源分组累加父任务输出 ----
        source_groups = defaultdict(int)
        if not is_entry:
            for pred_id in preds:
                pred_job = workflow.jobs[pred_id]
                if pred_job.state in ["执行完成", "已完成"] and pred_job.output_location is not None:
                    data_source = pred_job.output_location
                    source_groups[data_source] += pred_job.output_size

        EarliestFinishTime = []

        # ---- 遍历所有服务器 ----
        for server in env.servers:
            max_arrival_time = env.current_time

            if is_entry:
                # 入口任务：输入数据从设备上传
                device_source = ("MobileDevice", select_job.sourceDeviceId)
                if (server.server_type, server.id) != device_source:
                    bandwidth = env.get_bandwidth_between(
                        "MobileDevice", select_job.sourceDeviceId,
                        server.server_type, server.id
                    )
                    if bandwidth > 0:
                        trans_time = select_job.input_size / bandwidth
                        arrival = env.current_time + trans_time
                        max_arrival_time = max(max_arrival_time, arrival)
            else:
                # 非入口任务：按源分组累加父任务输出
                for data_source, total_size in source_groups.items():
                    if (server.server_type, server.id) == data_source:
                        continue
                    bandwidth = env.get_bandwidth_between(
                        data_source[0], data_source[1],
                        server.server_type, server.id
                    )
                    if bandwidth > 0:
                        trans_time = total_size / bandwidth
                        arrival = env.current_time + trans_time
                        max_arrival_time = max(max_arrival_time, arrival)

            current_transfer_time = max_arrival_time - env.current_time
            arrival_deadline = max_arrival_time

            # 统计队列任务
            jobs = server.wait_list + [
                tj for tj in server.trans_list
                if tj.arriveTime is not None and tj.arriveTime < arrival_deadline
            ]

            available_time = (
                server.processing_job.processEndTime - env.current_time
                if server.isBusy else 0
            )
            total_processing_time = sum(job.workload / server.pa for job in jobs)

            # 出口任务才加回传时间
            if is_exit:
                time_of_this_job = (
                    select_job.output_size / server.bandwidth
                    + select_job.workload / server.pa
                )
            else:
                time_of_this_job = select_job.workload / server.pa

            start_time = max(available_time + total_processing_time, current_transfer_time)
            EarliestFinishTime.append(start_time + time_of_this_job)

        # ---- 本地设备 ----
        device = env.devices[select_job.sourceDeviceId]
        max_arrival_time = env.current_time

        if is_entry:
            # 入口任务：输入数据在本地，不需要传输
            pass
        else:
            # 非入口任务：按源分组累加父任务输出
            for data_source, total_size in source_groups.items():
                if ("MobileDevice", device.id) == data_source:
                    continue
                bandwidth = env.get_bandwidth_between(
                    data_source[0], data_source[1],
                    "MobileDevice", device.id
                )
                if bandwidth > 0:
                    trans_time = total_size / bandwidth
                    arrival = env.current_time + trans_time
                    max_arrival_time = max(max_arrival_time, arrival)

        current_transfer_time = max_arrival_time - env.current_time

        available_time = (
            device.processing_job.processEndTime - env.current_time
            if device.isBusy else 0
        )
        total_processing_time = sum(job.workload / device.pa for job in device.wait_list)
        start_time = max(available_time + total_processing_time, current_transfer_time)
        time_of_this_job = select_job.workload / device.pa
        EarliestFinishTime.append(start_time + time_of_this_job)

        action = EarliestFinishTime.index(min(EarliestFinishTime))
        return action
    else:
        server = instance
        UR = []
        for job in server.wait_list:
            UR.append(job.UpwardRank)
        action = UR.index(max(UR))
        return action


def HEFT(env, instance, isR):
    """HEFT 启发式（完整版，考虑任务优先级插入）"""
    if isR:
        select_job = instance
        workflow = env.workflows[select_job.workflow_id]
        preds = list(workflow.graph.predecessors(select_job.id))
        is_entry = (len(preds) == 0)
        is_exit = (len(list(workflow.graph.successors(select_job.id))) == 0)

        # ---- 非入口任务：按源分组累加父任务输出 ----
        source_groups = defaultdict(int)
        if not is_entry:
            for pred_id in preds:
                pred_job = workflow.jobs[pred_id]
                if pred_job.state in ["执行完成", "已完成"] and pred_job.output_location is not None:
                    data_source = pred_job.output_location
                    source_groups[data_source] += pred_job.output_size

        EarliestFinishTime = []

        # ---- 遍历所有服务器 ----
        for server in env.servers:
            max_arrival_time = env.current_time

            if is_entry:
                # 入口任务：输入数据从设备上传
                device_source = ("MobileDevice", select_job.sourceDeviceId)
                if (server.server_type, server.id) != device_source:
                    bandwidth = env.get_bandwidth_between(
                        "MobileDevice", select_job.sourceDeviceId,
                        server.server_type, server.id
                    )
                    if bandwidth > 0:
                        trans_time = select_job.input_size / bandwidth
                        arrival = env.current_time + trans_time
                        max_arrival_time = max(max_arrival_time, arrival)
            else:
                # 非入口任务：按源分组累加父任务输出
                for data_source, total_size in source_groups.items():
                    if (server.server_type, server.id) == data_source:
                        continue
                    bandwidth = env.get_bandwidth_between(
                        data_source[0], data_source[1],
                        server.server_type, server.id
                    )
                    if bandwidth > 0:
                        trans_time = total_size / bandwidth
                        arrival = env.current_time + trans_time
                        max_arrival_time = max(max_arrival_time, arrival)

            current_transfer_time = max_arrival_time - env.current_time
            arrival_deadline = max_arrival_time

            # 统计队列任务
            jobs = server.wait_list + [
                tj for tj in server.trans_list
                if tj.arriveTime is not None and tj.arriveTime < arrival_deadline
            ]

            # 按 UpwardRank 降序排列，插入 select_job
            sorted_jobs = sorted(jobs, key=lambda j: j.UpwardRank, reverse=True)
            inserted = False
            for idx, job in enumerate(sorted_jobs):
                if select_job.UpwardRank > job.UpwardRank:
                    sorted_jobs.insert(idx, select_job)
                    inserted = True
                    break
            if not inserted:
                sorted_jobs.append(select_job)

            job_index = sorted_jobs.index(select_job)

            available_time = (
                server.processing_job.processEndTime - env.current_time
                if server.isBusy else 0
            )
            total_processing_time = sum(
                j.workload / server.pa for j in sorted_jobs[:job_index]
            )

            if is_exit:
                time_of_this_job = (
                    select_job.output_size / server.bandwidth
                    + select_job.workload / server.pa
                )
            else:
                time_of_this_job = select_job.workload / server.pa

            start_time = max(available_time + total_processing_time, current_transfer_time)
            EarliestFinishTime.append(start_time + time_of_this_job)

        # ---- 本地设备 ----
        device = env.devices[select_job.sourceDeviceId]
        max_arrival_time = env.current_time

        if is_entry:
            # 入口任务：输入数据在本地，不需要传输
            pass
        else:
            # 非入口任务：按源分组累加父任务输出
            for data_source, total_size in source_groups.items():
                if ("MobileDevice", device.id) == data_source:
                    continue
                bandwidth = env.get_bandwidth_between(
                    data_source[0], data_source[1],
                    "MobileDevice", device.id
                )
                if bandwidth > 0:
                    trans_time = total_size / bandwidth
                    arrival = env.current_time + trans_time
                    max_arrival_time = max(max_arrival_time, arrival)

        current_transfer_time = max_arrival_time - env.current_time

        # 按 UpwardRank 降序排列，插入 select_job
        jobs = device.wait_list
        sorted_jobs = sorted(jobs, key=lambda j: j.UpwardRank, reverse=True)
        inserted = False
        for idx, job in enumerate(sorted_jobs):
            if select_job.UpwardRank > job.UpwardRank:
                sorted_jobs.insert(idx, select_job)
                inserted = True
                break
        if not inserted:
            sorted_jobs.append(select_job)

        job_index = sorted_jobs.index(select_job)

        available_time = (
            device.processing_job.processEndTime - env.current_time
            if device.isBusy else 0
        )
        total_processing_time = sum(
            j.workload / device.pa for j in sorted_jobs[:job_index]
        )
        start_time = max(available_time + total_processing_time, current_transfer_time)
        time_of_this_job = select_job.workload / device.pa
        EarliestFinishTime.append(start_time + time_of_this_job)

        action = EarliestFinishTime.index(min(EarliestFinishTime))
        return action
    else:
        server = instance
        UR = []
        for job in server.wait_list:
            UR.append(job.UpwardRank)
        action = UR.index(max(UR))
        return action


def FCFS(env, instance, isR):
    """FCFS 启发式"""
    if isR:
        select_job = instance
        FirstIdleTime = []
        for server in env.servers:
            time_of_current_job = (
                server.processing_job.processEndTime - env.current_time
                if server.isBusy else 0
            )
            FirstIdleTime.append(time_of_current_job)
        device = env.devices[select_job.sourceDeviceId]
        time_of_current_job = (
            device.processing_job.processEndTime - env.current_time
            if device.isBusy else 0
        )
        FirstIdleTime.append(time_of_current_job)
        action = FirstIdleTime.index(min(FirstIdleTime))
        return action
    else:
        server = instance
        rt = []
        for job in server.wait_list:
            rt.append(job.readyTime)
        action = rt.index(min(rt))
        return action


def MAXMIN(env, instance, isR):
    """MAX-MIN 启发式"""
    if isR:
        select_job = instance
        FirstIdleTime = []
        for server in env.servers:
            time_of_current_job = (
                server.processing_job.processEndTime - env.current_time
                if server.isBusy else 0
            )
            FirstIdleTime.append(time_of_current_job)
        device = env.devices[select_job.sourceDeviceId]
        time_of_current_job = (
            device.processing_job.processEndTime - env.current_time
            if device.isBusy else 0
        )
        FirstIdleTime.append(time_of_current_job)
        action = FirstIdleTime.index(min(FirstIdleTime))
        return action
    else:
        server = instance
        pt = []
        for job in server.wait_list:
            pt.append(job.workload / server.pa)
        action = pt.index(max(pt))
        return action


def MINMIN(env, instance, isR):
    """MIN-MIN 启发式"""
    if isR:
        select_job = instance
        FirstIdleTime = []
        for server in env.servers:
            time_of_current_job = (
                server.processing_job.processEndTime - env.current_time
                if server.isBusy else 0
            )
            FirstIdleTime.append(time_of_current_job)
        device = env.devices[select_job.sourceDeviceId]
        time_of_current_job = (
            device.processing_job.processEndTime - env.current_time
            if device.isBusy else 0
        )
        FirstIdleTime.append(time_of_current_job)
        action = FirstIdleTime.index(min(FirstIdleTime))
        return action
    else:
        server = instance
        pt = []
        for job in server.wait_list:
            pt.append(job.workload / server.pa)
        action = pt.index(min(pt))
        return action