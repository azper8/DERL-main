import numpy as np


# 观察服务器,将任务分配到服务器上
def get_observe_R(env, job):
    server_info = []
    for server in env.servers:
        jobs = server.wait_list + [tj for tj in server.trans_list if tj.arriveTime < (env.current_time + job.data_up / server.bandwidth)]
        ft = server.processing_job.processEndTime - env.current_time if server.isBusy else 0
        wn = len(jobs)
        wp = sum(wj.workload / server.pa for wj in jobs)
        wd = sum(wj.data_down / server.bandwidth for wj in jobs)
        ut = job.data_up / server.bandwidth
        pt = job.workload / server.pa
        dt = job.data_down / server.bandwidth
        dl = env.workflows[job.workflow_id].deadline - env.current_time
        server_info.append([ft, wn, wp, wd, ut, pt, dt, dl])

    device = env.devices[job.sourceDeviceId]
    ft = device.processing_job.processEndTime - env.current_time if device.isBusy else 0
    wn = len(device.wait_list)
    wp = sum(wj.workload / device.pa for wj in device.wait_list)
    wd = 0
    ut = 0
    pt = job.workload / device.pa
    dt = 0
    dl = env.workflows[job.workflow_id].deadline - env.current_time
    server_info.append([ft, wn, wp, wd, ut, pt, dt, dl])
    server_info = np.array(server_info)
    return server_info


# 观察任务,选择任务执行
def get_observe_S(env, server):
    job_info = []
    for job in server.wait_list:
        mean_pa = sum(server.pa for server in env.servers) / len(env.servers)
        mean_bandwidth = sum(server.bandwidth for server in env.servers) / len(env.servers)
        workflow = env.workflows[job.workflow_id]
        pt = job.workload / server.pa
        dt = job.data_down / server.bandwidth if server.server_type != 'MobileDevice' else 0
        # pn = len(list(workflow.graph.predecessors(job.id)))
        sn = len(list(workflow.graph.successors(job.id)))
        cr = workflow.getCompleteRate(mean_pa, mean_bandwidth)
        # dr = job.DownwardRank
        ur = job.UpwardRank
        wt = env.current_time - job.readyTime
        ky = 1 if workflow.is_on_critical_path(job, mean_pa, mean_bandwidth) else 0  # 是否关键路径
        dl = env.workflows[job.workflow_id].deadline - env.current_time
        job_info.append([pt, dt, sn, cr, ur, wt, ky, dl])
    job_info = np.array(job_info)
    return job_info


def get_observe_R2(env, job):
    obs = []
    # 任务
    mean_pa = sum(server.pa for server in env.servers) / len(env.servers)
    mean_bandwidth = sum(server.bandwidth for server in env.servers) / len(env.servers)
    workflow = env.workflows[job.workflow_id]
    ul = job.data_up
    wl = job.workload
    dl = job.data_down
    pn = len(list(workflow.graph.predecessors(job.id)))
    sn = len(list(workflow.graph.successors(job.id)))
    cr = workflow.getCompleteRate()
    ur = job.UpwardRank
    wt = env.current_time - job.readyTime
    ky = 1 if workflow.is_on_critical_path(job, mean_pa, mean_bandwidth) else 0  # 是否关键路径
    obs.append([ul, wl, dl, pn, sn, cr, ur, wt, ky])
    # 服务器
    for server in env.servers:
        jobs = server.wait_list + [job for job in server.trans_list if job.arriveTime < (env.current_time + job.data_up / server.bandwidth)]
        ft = server.processing_job.finishTime - env.current_time if server.isBusy else 0
        wn = len(jobs)
        wl = sum([wj.workload for wj in jobs])
        bw = 1 / server.bandwidth
        pa = 1 / server.pa
        obs.append([ft, wn, wl, bw, pa])
    device = env.devices[job.sourceDeviceId]
    ft = device.processing_job.finishTime - env.current_time if device.isBusy else 0
    wn = len(device.wait_list)
    wl = sum([wj.workload for wj in device.wait_list])
    bw = 0
    pa = 1 / device.pa
    obs.append([ft, wn, wl, bw, pa, 0, 0, 0, 0])
    obs = np.array(obs)
    return obs


def get_observe_S2(env, server):
    obs = []
    # 服务器
    bw = 1 / server.bandwidth if server.server_type != 'MobileDevice' else 0
    pa = 1 / server.pa
    jobs = server.wait_list + [job for job in server.trans_list if
                               job.arriveTime < (env.current_time + job.data_up / server.bandwidth)] if server.server_type != 'MobileDevice' else server.wait_list
    wn = len(jobs)
    wl = sum([wj.workload / server.pa for wj in jobs])
    ur = sum([rj.processEndTime - rj.processStartTime for rj in server.record]) / env.current_time
    obs.append([bw, pa, wn, wl, ur])
    # 任务
    for job in server.wait_list:
        mean_pa = sum(server.pa for server in env.servers) / len(env.servers)
        mean_bandwidth = sum(server.bandwidth for server in env.servers) / len(env.servers)
        workflow = env.workflows[job.workflow_id]
        wl = job.workload
        pn = len(list(workflow.graph.predecessors(job.id)))
        sn = len(list(workflow.graph.successors(job.id)))
        cr = workflow.getCompleteRate()
        ur = job.UpwardRank
        wt = env.current_time - job.readyTime
        ky = 1 if workflow.is_on_critical_path(job, mean_pa, mean_bandwidth) else 0  # 是否关键路径
        obs.append([wl, pn, sn, cr, ur, wt, ky])
    obs = np.array(obs)
    return obs
