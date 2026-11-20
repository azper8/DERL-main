import numpy as np
from collections import defaultdict


def get_observe_R(env, job):
    """
    路由状态：为 job 选择最佳服务器/设备

    每个候选设备/服务器的状态向量 (8维)：
        sd1: ft / denom      - 设备剩余忙碌时间占比
        sd2: wn / mwn        - 队列长度占最大队列长度比例
        sd3: wp / mwp        - 队列工作量占最大队列工作量比例
        sd4: wd / denom      - 预计等待时间占比
        sd5: ut / denom      - 传输时间占比
        sd6: pt / denom      - 执行时间占比
        sd7: dt / denom      - 出口回传时间占比
        sd8: dl / budget     - 剩余截止时间占工作流预算比例

    其中：
        denom = max over all servers/devices of (max(ft, ut) + pt + dt)
        mwn   = max over all servers/devices of queue length
        mwp   = max over all servers/devices of queue workload
        budget = workflow.deadline - workflow.arrivalTime

    入口任务：输入数据从设备上传（input_size）
    非入口任务：按源分组累加父任务输出（output_size）
    """
    server_info = []

    workflow = env.workflows[job.workflow_id]
    preds = list(workflow.graph.predecessors(job.id))
    is_entry = (len(preds) == 0)
    is_exit = (len(list(workflow.graph.successors(job.id))) == 0)

    transfer_start = env.current_time

    # ---- 非入口任务：按源分组累加父任务输出 ----
    source_groups = defaultdict(int)
    if not is_entry:
        for pred_id in preds:
            pred_job = workflow.jobs[pred_id]
            if pred_job.state in ["执行完成", "已完成"] and pred_job.output_location is not None:
                source_groups[pred_job.output_location] += pred_job.output_size

    # ---- 工作流预算 ----
    budget = workflow.deadline - workflow.arrivalTime if workflow.deadline is not None else 1.0
    budget = max(1e-6, budget)

    # ---- 第一遍：计算所有服务器/设备的原始量 ----
    all_candidates = []  # [(ft, wn, wp, wd, ut, pt, dt), ...]

    for server in env.servers:
        ft = server.processing_job.processEndTime - env.current_time if server.isBusy else 0

        # ---- 传输时间 ----
        max_arrival_time = transfer_start

        if is_entry:
            # 入口任务：输入数据从设备上传
            device_source = ("MobileDevice", job.sourceDeviceId)
            if (server.server_type, server.id) != device_source:
                bandwidth = env.get_bandwidth_between(
                    "MobileDevice", job.sourceDeviceId,
                    server.server_type, server.id
                )
                if bandwidth > 0:
                    trans_time = job.input_size / bandwidth
                    arrival = transfer_start + trans_time
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
                    arrival = transfer_start + trans_time
                    max_arrival_time = max(max_arrival_time, arrival)

        ut = max_arrival_time - env.current_time
        arrival_deadline = max_arrival_time

        jobs = server.wait_list + [
            tj for tj in server.trans_list
            if tj.arriveTime is not None and tj.arriveTime < arrival_deadline
        ]
        wn = len(jobs)
        wp = sum(wj.workload / server.pa for wj in jobs)

        server_free_time = server.processing_job.processEndTime if server.isBusy else env.current_time
        wd = max(arrival_deadline, server_free_time) - env.current_time

        pt = job.workload / server.pa

        if is_exit and server.server_type != "MobileDevice":
            dt = job.output_size / server.bandwidth if server.bandwidth > 0 else 0
        else:
            dt = 0

        all_candidates.append((ft, wn, wp, wd, ut, pt, dt))

    # ---- 本地设备 ----
    device = env.devices[job.sourceDeviceId]
    ft = device.processing_job.processEndTime - env.current_time if device.isBusy else 0

    max_arrival_time = transfer_start

    if is_entry:
        # 入口任务：输入数据在本地，不需要传输
        pass
    else:
        for data_source, total_size in source_groups.items():
            if ("MobileDevice", device.id) == data_source:
                continue
            bandwidth = env.get_bandwidth_between(
                data_source[0], data_source[1],
                "MobileDevice", device.id
            )
            if bandwidth > 0:
                trans_time = total_size / bandwidth
                arrival = transfer_start + trans_time
                max_arrival_time = max(max_arrival_time, arrival)

    ut = max_arrival_time - env.current_time
    arrival_deadline = max_arrival_time

    jobs = device.wait_list + [
        tj for tj in device.trans_list
        if tj.arriveTime is not None and tj.arriveTime < arrival_deadline
    ]
    wn = len(jobs)
    wp = sum(wj.workload / device.pa for wj in jobs)

    device_free_time = device.processing_job.processEndTime if device.isBusy else env.current_time
    wd = max(arrival_deadline, device_free_time) - env.current_time

    pt = job.workload / device.pa
    dt = 0

    all_candidates.append((ft, wn, wp, wd, ut, pt, dt))

    # ---- 计算统一分母 ----
    denoms = [max(ft, ut) + pt + dt for (ft, wn, wp, wd, ut, pt, dt) in all_candidates]
    denom = max(denoms) + 1e-6

    mwn = max(wn for (ft, wn, wp, wd, ut, pt, dt) in all_candidates)
    mwn = max(1, mwn)

    mwp = max(wp for (ft, wn, wp, wd, ut, pt, dt) in all_candidates)
    mwp = max(1e-6, mwp)

    # ---- 第二遍：归一化 ----
    dl = workflow.deadline - env.current_time if workflow.deadline is not None else 0

    for (ft, wn, wp, wd, ut, pt, dt) in all_candidates:
        sd1 = ft / denom
        sd2 = wn / mwn
        sd3 = wp / mwp
        sd4 = wd / denom
        sd5 = ut / denom
        sd6 = pt / denom
        sd7 = dt / denom
        sd8 = dl / budget

        server_info.append([sd1, sd2, sd3, sd4, sd5, sd6, sd7, sd8])

    return np.array(server_info)


def get_observe_S(env, server):
    """
    每个候选任务的状态向量 (8维)：
        st1: pt / avg_budget          - 执行时间占平均预算比例
        st2: dt / avg_budget          - 回传时间占平均预算比例
        st3: sn / max_children        - 子任务数占工作流最大子任务数比例
        st4: cr                        - 工作流完成率
        st5: ur / max_upward_rank     - UpwardRank 占工作流最大 UpwardRank 比例
        st6: wt / avg_budget          - 等待时间占平均预算比例
        st7: ky                        - 是否在关键路径上
        st8: dl / (dl + pt + dt)      - 截止时间充裕度（紧急程度）
    """
    job_info = []
    mean_pa = sum(s.pa for s in env.servers) / len(env.servers)
    mean_bw = sum(s.bandwidth for s in env.servers) / len(env.servers)

    for idx, job in enumerate(server.wait_list):
        workflow = env.workflows[job.workflow_id]
        is_exit = (len(list(workflow.graph.successors(job.id))) == 0)

        # ---- 平均预算 ----
        budget = workflow.deadline - workflow.arrivalTime if workflow.deadline is not None else 1.0
        avg_budget = budget / workflow.num

        # ---- 原始量 ----
        pt = job.workload / server.pa

        if is_exit and server.server_type != "MobileDevice":
            dt = job.output_size / server.bandwidth if server.bandwidth > 0 else 0
        else:
            dt = 0

        sn = len(list(workflow.graph.successors(job.id)))
        max_children = max(
            len(list(workflow.graph.successors(j.id))) for j in workflow.jobs
        )
        max_children = max(1, max_children)

        cr = workflow.getCompleteRate()
        ur = job.UpwardRank if job.UpwardRank is not None else 0
        max_ur = max(j.UpwardRank for j in workflow.jobs if j.UpwardRank is not None)

        wt = env.current_time - job.readyTime if job.readyTime is not None else 0
        ky = 1 if workflow.is_on_critical_path(job, mean_pa, mean_bw) else 0
        dl = workflow.deadline - env.current_time if workflow.deadline is not None else 0

        # ---- 相对化 ----
        st1 = pt / avg_budget
        st2 = dt / avg_budget
        st3 = sn / max_children
        st4 = cr
        st5 = ur / max_ur
        st6 = wt / avg_budget
        st7 = ky
        st8 = dl / (dl + pt + dt + 1e-6)

        job_info.append([st1, st2, st3, st4, st5, st6, st7, st8])

    return np.array(job_info)