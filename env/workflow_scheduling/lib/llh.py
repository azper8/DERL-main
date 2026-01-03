def HH(env, instance, isR):
    if isR:
        select_job = instance
        EarliestFinishTime = []
        for server in env.servers:
            trans_job = [job for job in server.trans_list if job.arriveTime < (env.current_time + select_job.data_up / server.bandwidth)]
            jobs = server.wait_list + trans_job
            available_time = server.processing_job.processEndTime - env.current_time if server.isBusy else 0
            total_processing_time = sum(job.workload for job in jobs) / server.pa
            start_time = max(available_time + total_processing_time, select_job.data_up / server.bandwidth)
            time_of_this_job = select_job.data_down / server.bandwidth + select_job.workload / server.pa
            EarliestFinishTime.append(start_time + time_of_this_job)
        device = env.devices[select_job.sourceDeviceId]
        available_time = device.processing_job.processEndTime - env.current_time if device.isBusy else 0
        total_processing_time = sum(job.workload for job in device.wait_list) / device.pa
        start_time = available_time + total_processing_time
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
    if isR:
        select_job = instance
        EarliestFinishTime = []
        for server in env.servers:
            readyTime_job = select_job.readyTime + select_job.data_up / server.bandwidth
            readyTime_server = server.processing_job.processEndTime if server.isBusy else env.current_time
            readyTime = max(readyTime_job, readyTime_server)
            finishTime = readyTime + select_job.workload / server.pa
            EarliestFinishTime.append(finishTime)
        device = env.devices[select_job.sourceDeviceId]
        readyTime = device.processing_job.processEndTime if device.isBusy else env.current_time
        finishTime = readyTime + select_job.workload / device.pa
        EarliestFinishTime.append(finishTime)
        # EarliestFinishTime = np.array(EarliestFinishTime)
        # min_val = np.min(EarliestFinishTime)
        # min_indices = np.where(EarliestFinishTime == min_val)[0]
        # action = np.random.choice(min_indices)
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
    if isR:
        select_job = instance
        FirstIdleTime = []
        for server in env.servers:
            time_of_current_job = server.processing_job.processEndTime - env.current_time if server.isBusy else 0
            FirstIdleTime.append(time_of_current_job)
        device = env.devices[select_job.sourceDeviceId]
        time_of_current_job = device.processing_job.processEndTime - env.current_time if device.isBusy else 0
        FirstIdleTime.append(time_of_current_job)
        # FirstIdleTime = np.array(FirstIdleTime)
        # min_val = np.min(FirstIdleTime)
        # min_indices = np.where(FirstIdleTime == min_val)[0]
        # action = np.random.choice(min_indices)
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
    if isR:
        select_job = instance
        FirstIdleTime = []
        for server in env.servers:
            time_of_current_job = server.processing_job.processEndTime - env.current_time if server.isBusy else 0
            FirstIdleTime.append(time_of_current_job)
        device = env.devices[select_job.sourceDeviceId]
        time_of_current_job = device.processing_job.processEndTime - env.current_time if device.isBusy else 0
        FirstIdleTime.append(time_of_current_job)
        # FirstIdleTime = np.array(FirstIdleTime)
        # min_val = np.min(FirstIdleTime)
        # min_indices = np.where(FirstIdleTime == min_val)[0]
        # action = np.random.choice(min_indices)
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
    if isR:
        select_job = instance
        FirstIdleTime = []
        for server in env.servers:
            time_of_current_job = server.processing_job.processEndTime - env.current_time if server.isBusy else 0
            FirstIdleTime.append(time_of_current_job)
        device = env.devices[select_job.sourceDeviceId]
        time_of_current_job = device.processing_job.processEndTime - env.current_time if device.isBusy else 0
        FirstIdleTime.append(time_of_current_job)
        # FirstIdleTime = np.array(FirstIdleTime)
        # min_val = np.min(FirstIdleTime)
        # min_indices = np.where(FirstIdleTime == min_val)[0]
        # action = np.random.choice(min_indices)
        action = FirstIdleTime.index(min(FirstIdleTime))
        return action
    else:
        server = instance
        pt = []
        for job in server.wait_list:
            pt.append(job.workload / server.pa)
        action = pt.index(min(pt))
        return action


'''def SDLS(env,instance,isR):
    if isR:
        select_job = instance
        SDL = []
        workflow = env.workflows[select_job.workflow_id]
        mean_pa = sum(server.pa for server in env.servers) / len(env.servers)
        mean_bandwidth = sum(server.bandwidth for server in env.servers) / len(env.servers)
        workflow.calculate_sb_level(mean_pa, mean_bandwidth)
        sbLevel = select_job.SbLevel
        for server in env.servers:
            readyTime = server.processing_job.endTime if server.isBusy else select_job.readyTime
            upload = select_job.data_up
            workload = select_job.workload
            download = select_job.data_down
            time_of_selected_job = (upload + download) / server.bandwidth + workload / server.pa
            SDL.append(sbLevel+readyTime+time_of_selected_job)
        device = env.devices[select_job.sourceDeviceId]
        readyTime = device.processing_job.endTime if device.isBusy else select_job.readyTime
        upload = select_job.data_up
        workload = select_job.workload
        download = select_job.data_down
        time_of_selected_job = (upload + download) / device.bandwidth + workload / device.pa
        SDL.append(sbLevel+readyTime+time_of_selected_job)
        action = SDL.index(max(SDL))
        return action
    else:
        server = instance
        sb = []
        for job in server.wait_list:
            workflow = env.workflows[job.workflow_id]
            mean_pa = sum(server.pa for server in env.servers) / len(env.servers)
            mean_bandwidth = sum(server.bandwidth for server in env.servers) / len(env.servers)
            workflow.calculate_sb_level(mean_pa, mean_bandwidth)
            sb.append(job.SbLevel)
        action = sb.index(max(sb))
        return action'''

def BWAWA(env,instance,isR):
    if isR:
        select_job = instance
        TransmissionTime = []
        for server in env.servers:
            TransmissionTime.append(select_job.data_up / server.bandwidth)
        TransmissionTime.append(0)
        action = TransmissionTime.index(min(TransmissionTime))
        return action
    else:
        server = instance
        DR = []
        for job in server.wait_list:
            DR.append(job.DownwardRank)
        action = DR.index(min(DR))
        return action
