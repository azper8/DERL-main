from copy import deepcopy


def get_GP_Priority_RA(env, selected_job, func_RA):
    score = []
    for server in env.servers:
        jobs = server.wait_list + [job for job in server.trans_list if job.arriveTime < (env.current_time + job.data_up / server.bandwidth)]
        NIQ = len(jobs)
        WIQ = sum(job.workload / server.pa for job in jobs)
        MRT = server.processing_job.processEndTime - env.current_time if server.isBusy else 0
        UT = selected_job.data_up / server.bandwidth
        DT = selected_job.data_down / server.bandwidth
        PT = selected_job.workload / server.pa
        # TTIQ = sum((job.data_up + job.data_down) / server.bandwidth + job.workload / server.pa for job in jobs)
        TTIQ = env.workflows[selected_job.workflow_id].deadline - env.current_time
        TIS = env.current_time - env.workflows[selected_job.workflow_id].arrivalTime
        TWT = 0
        NTR = len([job for job in env.workflows[selected_job.workflow_id].jobs if job.state != "已完成"])
        # DDL = env.workflows[selected_job.workflow_id].deadline - env.current_time
        result_RA = func_RA(NIQ, WIQ, MRT, UT, DT, PT, TTIQ, TIS, TWT, NTR)
        score.append(result_RA)

    device = env.devices[selected_job.sourceDeviceId]
    NIQ = len(device.wait_list)
    WIQ = sum(job.workload / device.pa for job in device.wait_list)
    MRT = device.processing_job.processEndTime - env.current_time if device.isBusy else 0
    UT = 0
    DT = 0
    PT = selected_job.workload / device.pa
    # TTIQ = sum(job.workload / device.pa for job in device.wait_list)
    TTIQ = env.workflows[selected_job.workflow_id].deadline - env.current_time
    TIS = env.current_time - env.workflows[selected_job.workflow_id].arrivalTime
    TWT = 0
    NTR = len([job for job in env.workflows[selected_job.workflow_id].jobs if job.state != "已完成"])
    # DDL = env.workflows[selected_job.workflow_id].deadline - env.current_time
    result_RA = func_RA(NIQ, WIQ, MRT, UT, DT, PT, TTIQ, TIS, TWT, NTR)
    score.append(result_RA)

    index = score.index(max(score))
    return index


def get_GP_Priority_SA(env, server, func_SA):
    score = []
    for job in server.wait_list:
        NIQ = len(server.wait_list)
        WIQ = sum(job.workload / server.pa for job in server.wait_list)
        MRT = server.processing_job.processEndTime - env.current_time if server.isBusy else 0
        UT = job.data_up / server.bandwidth if server.server_type != 'MobileDevice' else 0
        DT = job.data_down / server.bandwidth if server.server_type != 'MobileDevice' else 0
        PT = job.workload / server.pa
        # if server.server_type == 'MobileDevice':
        #     TTIQ = sum(job.workload / server.pa for job in server.wait_list)
        # else:
        #     TTIQ = sum((job.data_up + job.data_down) / server.bandwidth + job.workload / server.pa for job in server.wait_list)
        TTIQ = env.workflows[job.workflow_id].deadline - env.current_time
        TIS = env.current_time - env.workflows[job.workflow_id].arrivalTime
        TWT = env.current_time - job.readyTime
        NTR = len([job for job in env.workflows[job.workflow_id].jobs if job.state != "已完成"])
        # DDL = env.workflows[job.workflow_id].deadline - env.current_time
        result_SA = func_SA(NIQ, WIQ, MRT, UT, DT, PT, TTIQ, TIS, TWT, NTR)
        score.append(result_SA)

        index = score.index(max(score))
        return index
