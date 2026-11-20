# network/reuse.py
"""
数据复用层（新版模型）

职责:
  1. 从 DAX XML 解析 workflow（复用 env.lib.readXML）
  2. 构造 Job / Server / Workflow 对象（复用 env.lib.*）
  3. 复刻 CreateDataset 的 timeTable / 资源生成 / deadline 计算逻辑
  4. 提供静态带宽查询（复刻 SimEnv.get_bandwidth_between，去掉波动）
  5. 提供 job snapshot 序列化（跨节点传输用）

新模型要点:
  - Job 用 input_size / output_size（标量），没有文件字典
  - 传输：同源累加 output_size，异源取 max
  - 非入口任务的 input_size 不参与传输
  - deadline 用固定参考环境算 HEFT baseline

对外入口:
  - build_resources(seed)  → (devices, servers)            # Server 节点用
  - build_all(seed)        → (devices, servers, workflows) # Master / Mobile 用
"""
import os
import copy
import random
import heapq
import pickle

import numpy as np
import networkx as nx

from env.lib.job import Job
from env.lib.server import Server
from env.lib.workflow import Workflow
from env.lib.readXML import generate_workflow_from_xml, sample_task_io_by_topology
from env.lib.llh import HEFT
from env.lib.event import (
    jobAssignEvent, jobProcessEvent, workflowArriveEvent,
)
from env.simulator import SimEnv

from network.config import SCENARIO, PROJECT_ROOT


# ============================================================
#  DAX 文件映射（与 CreateDataset.switcher 一致）
# ============================================================
SWITCHER = {
    0: "CyberShake_30.xml",
    1: "Epigenomics_24.xml",
    2: "Ligo_30.xml",
    3: "Montage_25.xml",
    4: "Sipht_29.xml",
    5: "CyberShake_50.xml",
    6: "Epigenomics_47.xml",
    7: "Ligo_50.xml",
    8: "Montage_50.xml",
    9: "Sipht_58.xml",
    10: "CyberShake_100.xml",
    11: "Epigenomics_100.xml",
    12: "Ligo_100.xml",
    13: "Montage_100.xml",
    14: "Sipht_97.xml",
}

DAX_BASE_PATH = os.path.join(PROJECT_ROOT, "env", "dax")


def get_dag_type_path(dag_type):
    return os.path.join(DAX_BASE_PATH, SWITCHER.get(dag_type, "CyberShake_30.xml"))


def pick_dag_type(wf_size):
    if wf_size == "small":
        return random.randint(0, 4)
    elif wf_size == "medium":
        return random.randint(0, 9)
    elif wf_size == "large":
        return random.randint(0, 14)
    return random.randint(0, 4)


# ============================================================
#  资源生成
# ============================================================
def create_mobile_devices(num_mobile):
    mds = []
    for md_id in range(num_mobile):
        pa = random.uniform(*SCENARIO["mobile_pa_range"])
        name = f"MobileDevice_{md_id}"
        md = Server(md_id, None, pa, "MobileDevice", name)
        mds.append(md)
    return mds


def create_servers(num_edge, num_center):
    ess = []
    for ed_id in range(num_edge):
        bandwidth = SCENARIO["edge_bandwidth"]
        pa = random.uniform(*SCENARIO["edge_pa_range"])
        name = f"EdgeServer_{ed_id}"
        es = Server(ed_id, bandwidth, pa, "EdgeServer", name)
        es.region = 0
        ess.append(es)

    css = []
    bw_choices, bw_probs = SCENARIO["center_bandwidth_choices"]
    for cs_id in range(num_edge, num_edge + num_center):
        bandwidth_choice = np.random.choice(bw_choices, p=bw_probs)
        bandwidth = bandwidth_choice * 1024 * 1024
        pa = random.uniform(*SCENARIO["center_pa_range"])
        name = f"CenterServer_{cs_id - num_edge}"
        cs = Server(cs_id, bandwidth, pa, "CenterServer", name)
        cs.region = bandwidth_choice
        css.append(cs)

    return ess + css


# ============================================================
#  固定参考环境（deadline 计算专用，seed=41）
# ============================================================
def create_reference_servers(seed=41):
    rng = np.random.RandomState(seed)
    servers = []
    for i in range(10):
        pa = rng.uniform(1500, 3000)
        server = Server(
            es_id=i,
            bandwidth=1024 * 1024 * 1024,
            pa=pa,
            server_type="EdgeServer",
            name=f"RefEdge_{i}"
        )
        server.region = 0
        servers.append(server)
    for i in range(10):
        bw_mbps = rng.choice([128, 256, 512], p=[0.2, 0.6, 0.2])
        pa = rng.uniform(3000, 6000)
        server = Server(
            es_id=10 + i,
            bandwidth=bw_mbps * 1024 * 1024,
            pa=pa,
            server_type="CenterServer",
            name=f"RefCenter_{i}"
        )
        server.region = bw_mbps
        servers.append(server)
    return servers


def create_reference_devices(seed=41):
    rng = np.random.RandomState(seed)
    pa = rng.uniform(750, 1500)
    device = Server(
        es_id=0,
        bandwidth=None,
        pa=pa,
        server_type="MobileDevice",
        name=f"RefDevice_0"
    )
    return [device]


# ============================================================
#  工作流到达时间表
# ============================================================
def create_workflow_time_table(num_mobile, num_workflows, lambda_rate):
    timeTable = []
    for device_id in range(num_mobile):
        workflow_list = [0]
        for _ in range(1, num_workflows):
            interval = random.expovariate(lambda_rate)
            workflow_list.append(workflow_list[-1] + interval)
        timeTable.append(workflow_list)

    all_times = [(t, i) for i, sub in enumerate(timeTable) for t in sub]
    sorted_times = sorted(all_times, key=lambda x: x[0])
    return sorted_times[:num_workflows]


# ============================================================
#  HEFT 单工作流 baseline
# ============================================================
def run_heft_on_single_workflow(wf, servers, devices):
    wf.id = 0
    for job in wf.jobs:
        job.workflow_id = 0

    temp_env = SimEnv(devices, [wf], servers)

    for server in temp_env.servers:
        server.wait_list = []
        server.trans_list = []
        server.isBusy = False
        server.processing_job = None
    for device in temp_env.devices:
        device.wait_list = []
        device.trans_list = []
        device.isBusy = False
        device.processing_job = None
    temp_env.ready_job_set = set()
    temp_env.event_list = []
    temp_env.current_time = 0

    # mean_bw 排除 bandwidth=None 的 mobile
    server_list_with_bw = [s for s in servers if s.bandwidth is not None]
    mean_pa = sum(s.pa for s in servers) / len(servers) if servers else 1
    mean_bw = (
        sum(s.bandwidth for s in server_list_with_bw) / len(server_list_with_bw)
        if server_list_with_bw else 1
    )

    for job in wf.jobs:
        job.UpwardRank = 0.0
        job.DownwardRank = 0.0

    wf.calculate_upward_ranks(mean_pa, mean_bw)
    wf.calculate_downward_ranks(mean_pa, mean_bw)

    for job in wf.jobs:
        if job.UpwardRank is None:
            job.UpwardRank = 0.0
        if job.DownwardRank is None:
            job.DownwardRank = 0.0

    temp_env.addEvent(workflowArriveEvent(wf, None, None, wf.arrivalTime))

    current_time = 0
    while True:
        while temp_env.event_list and temp_env.event_list[0].time == current_time:
            event = heapq.heappop(temp_env.event_list)
            event.trigger(temp_env)

        if temp_env.ready_job_set:
            select_job = max(
                temp_env.ready_job_set,
                key=lambda x: x.UpwardRank if x.UpwardRank is not None else 0.0
            )
            action = HEFT(temp_env, select_job, isR=True)
            if action == len(temp_env.servers):
                temp_env.addEvent(jobAssignEvent(
                    temp_env.workflows[select_job.workflow_id],
                    select_job,
                    temp_env.devices[select_job.sourceDeviceId],
                    current_time))
            else:
                temp_env.addEvent(jobAssignEvent(
                    temp_env.workflows[select_job.workflow_id],
                    select_job,
                    temp_env.servers[action],
                    current_time))
            continue

        for device in temp_env.devices:
            if not device.isBusy and device.wait_list:
                action = HEFT(temp_env, device, isR=False)
                job = device.wait_list[action]
                temp_env.addEvent(jobProcessEvent(
                    temp_env.workflows[job.workflow_id], job, device, current_time))
                continue

        for server in temp_env.servers:
            if server.available and not server.isBusy and server.wait_list:
                action = HEFT(temp_env, server, isR=False)
                job = server.wait_list[action]
                temp_env.addEvent(jobProcessEvent(
                    temp_env.workflows[job.workflow_id], job, server, current_time))
                continue

        if not temp_env.event_list:
            break

        current_time = temp_env.event_list[0].time

    return wf.getWorkflowFinishTime()


# ============================================================
#  工作流生成（新版模型）
# ============================================================
def create_workflows(time_table, servers, devices, wf_size, probs=None):
    """
    time_table: [(arrival_time, device_id), ...]
    probs: deadline 系数 [2, 3] 的概率，默认 [0.8, 0.2]
    """
    if probs is None:
        probs = [0.8, 0.2]

    ref_servers = create_reference_servers(seed=41)
    ref_devices = create_reference_devices(seed=41)

    workflows = []
    wf_id = 0

    for release_time, device_id in time_table:
        dag_type = pick_dag_type(wf_size)
        dag_path = get_dag_type_path(dag_type)
        G, runtime_map, task_name_map = generate_workflow_from_xml(dag_path)
        if G is None:
            continue

        # 任务级 IO 采样（新版）
        input_map, output_map = sample_task_io_by_topology(
            G, target_min_mb=5220, target_max_mb=20680
        )

        num_jobs = G.number_of_nodes()
        workload_list = np.random.uniform(30000, 90000, size=num_jobs)

        jobs = []
        for task_id in range(num_jobs):
            job = Job(
                job_id=task_id,
                workflow_id=wf_id,
                input_size=input_map.get(task_id, 0),
                output_size=output_map.get(task_id, 0),
                workload=workload_list[task_id],
                sourceDeviceId=device_id,
            )
            jobs.append(job)

        wf = Workflow(wf_id, SWITCHER.get(dag_type, "Invalid"),
                      release_time, G, jobs, device_id)

        # 用固定参考环境算 deadline
        wf_for_heft = copy.deepcopy(wf)
        wf_for_heft.id = 0
        for job in wf_for_heft.jobs:
            job.workflow_id = 0
        wf_for_heft.sourceDeviceId = 0
        for job in wf_for_heft.jobs:
            job.sourceDeviceId = 0

        heft_makespan = run_heft_on_single_workflow(
            wf_for_heft, ref_servers, ref_devices
        )
        wf.deadline = wf.arrivalTime + heft_makespan * np.random.choice(
            [2, 3], p=probs)

        workflows.append(wf)
        wf_id += 1

    return workflows


# ============================================================
#  静态带宽查询
# ============================================================
def get_bandwidth_between(servers, devices, src_type, src_id, dst_type, dst_id):
    if src_type == dst_type and src_id == dst_id:
        return float('inf')

    lookup = {}
    for d in devices:
        lookup[(d.server_type, d.id)] = d
    for s in servers:
        lookup[(s.server_type, s.id)] = s

    src = lookup.get((src_type, src_id))
    dst = lookup.get((dst_type, dst_id))
    if src is None or dst is None:
        return 0

    if src_type == "MobileDevice" and dst_type != "MobileDevice":
        return dst.bandwidth
    if dst_type == "MobileDevice" and src_type != "MobileDevice":
        return src.bandwidth
    if src_type == "MobileDevice" and dst_type == "MobileDevice":
        return 0

    if src_type == "EdgeServer" and dst_type == "EdgeServer":
        return 1024 * 1024 * 1024

    if src_type == "CenterServer" and dst_type == "CenterServer":
        r1, r2 = src.region, dst.region
        if r1 == r2:
            bw = 1024
        elif (r1 == 128 and r2 == 256) or (r1 == 256 and r2 == 128) or \
             (r1 == 256 and r2 == 512) or (r1 == 512 and r2 == 256):
            bw = 512
        else:
            bw = 256
        return bw * 1024 * 1024

    if (src_type == "EdgeServer" and dst_type == "CenterServer") or \
       (src_type == "CenterServer" and dst_type == "EdgeServer"):
        cloud = dst if dst_type == "CenterServer" else src
        return cloud.region * 1024 * 1024

    return 1024 * 1024 * 1024


# ============================================================
#  workflow 分组
# ============================================================
def group_workflows_by_device(workflows, num_mobile):
    groups = {i: [] for i in range(num_mobile)}
    for wf in workflows:
        groups[wf.sourceDeviceId].append(wf)
    return groups


# ============================================================
#  job snapshot
# ============================================================
def make_job_snapshot(job):
    """新模型：只有 input_size / output_size"""
    return {
        "workload":      float(job.workload),
        "input_size":    int(job.input_size),
        "output_size":   int(job.output_size),
        "source_device": f"MobileDevice_{job.sourceDeviceId}",
    }


def job_from_snapshot(job_id, workflow_id, snapshot):
    source_id_str = snapshot.get("source_device", "MobileDevice_0")
    source_id = int(source_id_str.split("_")[-1])

    job = Job(
        job_id=job_id,
        workflow_id=workflow_id,
        input_size=snapshot.get("input_size", 0),
        output_size=snapshot.get("output_size", 0),
        workload=snapshot["workload"],
        sourceDeviceId=source_id,
    )
    return job


# ============================================================
#  对外入口
# ============================================================
def build_resources(seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    num_mobile = SCENARIO["mobile_num"]
    num_edge   = SCENARIO["edge_num"]
    num_center = SCENARIO["center_num"]

    devices = create_mobile_devices(num_mobile)
    servers = create_servers(num_edge, num_center)
    return devices, servers


def build_all(seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    num_mobile  = SCENARIO["mobile_num"]
    num_edge    = SCENARIO["edge_num"]
    num_center  = SCENARIO["center_num"]
    num_wf      = SCENARIO["wf_num"]
    wf_size     = SCENARIO["wf_size"]
    lambda_rate = SCENARIO["lambda_rate"]

    devices = create_mobile_devices(num_mobile)
    servers = create_servers(num_edge, num_center)
    time_table = create_workflow_time_table(num_mobile, num_wf, lambda_rate)
    workflows = create_workflows(time_table, servers, devices, wf_size)

    return devices, servers, workflows