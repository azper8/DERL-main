# network/mobile_node.py
"""
MobileDevice 节点（新版模型）

关键改动:
  - 传输模型: 同源累加 output_size, 异源取 max; 入口任务上传 input_size
  - RA/SA 矩阵: 与训练时 observe.py 一致, 归一化后拼矩阵
  - job 只有 input_size / output_size
"""
import time
import logging
import threading
from collections import defaultdict

import numpy as np
from flask import request, jsonify

from network.config import (
    SCENARIO, NODES, RUNTIME, MASTER_URL,
)
from network.node_base import BaseNode
from network.reuse import (
    build_all, get_bandwidth_between, make_job_snapshot,
)
from network import communication
from network.protocol import (
    make_observe_ra_req, make_reserve_req, make_submit_req,
    make_notify_ready_resp, make_report_done_req,
)
from network.decision import get_decision

logger = logging.getLogger(__name__)


class MobileNode(BaseNode):
    def __init__(self, device_id, master_url=None):
        node_id = f"MobileDevice_{device_id}"
        super().__init__(node_id, "MobileDevice", master_url)
        self.device_id = device_id

        self.devices = None
        self.servers = None
        self.my_workflows = []
        self.server_map = {}
        self.device_map = {}
        self.url_of = {}

        # job 图状态
        self.job_lookup = {}
        self.job_parents = {}
        self.job_children = {}
        self.job_is_exit = {}
        self.job_is_entry = {}
        self.done_parents = {}
        self.job_output_location = {}
        self.job_process_end = {}

        # 调度状态
        self.ready_set = set()
        self.active_workflows = set()
        self.wf_by_id = {}

        # 本地执行
        self.local_wait_list = []
        self.local_processing = None
        self.local_is_busy = False
        self.local_busy_until = 0.0

        self.reported_done = False

        self.lock = threading.RLock()
        self._wakeup = threading.Event()
        self.decision = None

        self._register_business_routes()

    def _register_business_routes(self):
        @self.app.route("/observe_sa", methods=["POST"])
        def _observe_sa():
            body = request.get_json(force=True)
            server_id = body["server_id"]
            wf_id = body["workflow_id"]
            job_ids = body["job_ids"]
            resp = self._handle_observe_sa(server_id, wf_id, job_ids)
            return jsonify(resp)

        @self.app.route("/notify_ready", methods=["POST"])
        def _notify_ready():
            body = request.get_json(force=True)
            self._handle_notify_ready(body)
            return jsonify(make_notify_ready_resp())

    # ========================================================
    #  prepare
    # ========================================================
    def prepare(self):
        logger.info(f"[{self.node_id}] 加载数据 ...")
        seed = SCENARIO["seed"]

        devices, servers, workflows = build_all(seed=seed)
        self.devices = devices
        self.servers = servers

        for d in devices:
            self.device_map[d.name] = d
        for s in servers:
            self.server_map[s.name] = s

        self.url_of = dict(NODES)

        self.my_workflows = [wf for wf in workflows
                             if wf.sourceDeviceId == self.device_id]
        logger.info(f"[{self.node_id}] 分到 {len(self.my_workflows)} 个 workflow")

        for wf in self.my_workflows:
            self.wf_by_id[wf.id] = wf
            for job in wf.jobs:
                key = (wf.id, job.id)
                self.job_lookup[key] = job
                self.job_parents[key] = list(wf.graph.predecessors(job.id))
                self.job_children[key] = list(wf.graph.successors(job.id))
                self.job_is_exit[key] = (len(self.job_children[key]) == 0)
                self.job_is_entry[key] = (len(self.job_parents[key]) == 0)
                self.done_parents[key] = set()

        self.decision = get_decision()
        logger.info(f"[{self.node_id}] 决策器加载完成")

        self.ready = True
        logger.info(f"[{self.node_id}] prepare 完成")

    # ========================================================
    #  主循环
    # ========================================================
    def run_main_loop(self):
        while not self.shutdown_flag.is_set():
            try:
                self._activate_workflows()

                if self._try_process_one_ready_job():
                    continue

                if self._try_process_local_job():
                    continue

                if not self.reported_done and self._all_workflows_done():
                    self._report_done()
                    self.reported_done = True
            except Exception:
                logger.exception(f"[{self.node_id}] 主循环异常")

            self._wakeup.wait(timeout=RUNTIME["main_loop_sleep"])
            self._wakeup.clear()

    def _activate_workflows(self):
        now = time.time() - self.start_time
        with self.lock:
            for wf in self.my_workflows:
                if wf.id in self.active_workflows:
                    continue
                if now >= wf.arrivalTime:
                    self.active_workflows.add(wf.id)
                    for job in wf.jobs:
                        if len(self.job_parents[(wf.id, job.id)]) == 0:
                            self.ready_set.add((wf.id, job.id))
                            job.readyTime = wf.arrivalTime

    def _try_process_one_ready_job(self):
        with self.lock:
            if not self.ready_set:
                return False
            job_key = min(self.ready_set,
                          key=lambda k: self.job_lookup[k].workload)
            self.ready_set.discard(job_key)
        self._handle_ra(job_key)
        return True

    # ========================================================
    #  RA
    # ========================================================
    def _handle_ra(self, job_key):
        wf_id, jid = job_key
        job = self.job_lookup[job_key]
        wf = self.wf_by_id[wf_id]

        now = time.time() - self.start_time
        is_exit = self.job_is_exit[job_key]

        # ---- 本地算 ut/pt/dt ----
        ut_list, pt_list, dt_list = [], [], []
        for s in self.servers:
            ut = self._compute_ut(job, s.server_type, s.id, now)
            pt = job.workload / s.pa
            dt = (job.output_size / s.bandwidth) if is_exit else 0.0
            ut_list.append(ut); pt_list.append(pt); dt_list.append(dt)

        my_device = self.device_map[self.node_id]
        ut_self = 0.0
        pt_self = job.workload / my_device.pa
        dt_self = 0.0
        ut_list.append(ut_self); pt_list.append(pt_self); dt_list.append(dt_self)

        dl = wf.deadline - now

        # ---- fan-out observe_ra ----
        server_urls = [self.url_of[s.name] for s in self.servers]
        payloads = []
        for i, s in enumerate(self.servers):
            payloads.append(make_observe_ra_req(
                wf_id, jid,
                ut=ut_list[i], pt=pt_list[i], dt=dt_list[i], dl=dl))
        responses = self._fan_out_observe_ra(server_urls, payloads)

        # ---- 拼原始量 ----
        # 顺序: [server_0, ..., server_{N-1}, self]
        raw = np.zeros((len(self.servers) + 1, 7), dtype=np.float64)  # ft,wn,wp,wd,ut,pt,dt
        for i, (ok, data) in enumerate(responses):
            if not ok:
                raw[i] = [1e9, 1e9, 1e9, 1e9, ut_list[i], pt_list[i], dt_list[i]]
                continue
            raw[i] = [data["ft"], data["wn"], data["wp"], data["wd"],
                      ut_list[i], pt_list[i], dt_list[i]]

        # local
        with self.lock:
            local_wait = len(self.local_wait_list)
            local_wp = sum(self.job_lookup[k].workload / my_device.pa
                           for k in self.local_wait_list)
            local_ft = (self.local_busy_until - now) if self.local_is_busy else 0.0
            if local_ft < 0:
                local_ft = 0.0
        local_wd = max(ut_self, local_ft) - now if local_ft > ut_self else 0.0
        raw[-1] = [local_ft, local_wait, local_wp, local_wd,
                   ut_self, pt_self, dt_self]

        # ---- 归一化（与 observe.py 一致）----
        ft = raw[:, 0]; wn = raw[:, 1]; wp = raw[:, 2]; wd = raw[:, 3]
        ut = raw[:, 4]; pt = raw[:, 5]; dt = raw[:, 6]

        denoms = np.maximum(ft, ut) + pt + dt
        denom = float(denoms.max()) + 1e-6
        mwn = max(1, int(wn.max()))
        mwp = float(wp.max()) + 1e-6
        budget = max(1e-6, wf.deadline - wf.arrivalTime)

        matrix = np.zeros((len(self.servers) + 1, 8), dtype=np.float64)
        matrix[:, 0] = ft / denom
        matrix[:, 1] = wn / mwn
        matrix[:, 2] = wp / mwp
        matrix[:, 3] = wd / denom
        matrix[:, 4] = ut / denom
        matrix[:, 5] = pt / denom
        matrix[:, 6] = dt / denom
        matrix[:, 7] = dl / budget

        # ---- 决策 ----
        action = self.decision.decide_ra(matrix)

        # ---- 执行 ----
        if action == len(self.servers):
            with self.lock:
                self.local_wait_list.append(job_key)
            self._wakeup.set()
        else:
            target = self.servers[action]
            target_url = self.url_of[target.name]
            ut = ut_list[action]

            reserve_payload = make_reserve_req(
                wf_id, jid, ut, self.node_id, job.workload)
            communication.reserve(target_url, reserve_payload)

            snapshot = make_job_snapshot(job)
            submit_payload = make_submit_req(
                wf_id, jid, snapshot, self.node_id, is_exit)

            def _delayed_submit():
                communication.submit_job(target_url, submit_payload)

            if ut > 0:
                threading.Timer(ut, _delayed_submit).start()
            else:
                _delayed_submit()

    def _fan_out_observe_ra(self, urls, payloads):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results = [None] * len(urls)

        def _call(i):
            ok, data, err = communication.post(
                urls[i], "/observe_ra", payloads[i], timeout=30, retries=2)
            results[i] = (ok, data if ok else None)

        with ThreadPoolExecutor(max_workers=min(32, len(urls))) as ex:
            futures = [ex.submit(_call, i) for i in range(len(urls))]
            for _ in as_completed(futures):
                pass
        return results

    def _compute_ut(self, job, dst_type, dst_id, now):
        """
        新模型传输：
          入口任务: input_size / bw(src=mobile -> dst)
          非入口: 按源分组累加 output_size, 异源取 max
        """
        wf_id = job.workflow_id
        job_key = (wf_id, job.id)
        preds = self.job_parents[job_key]
        is_entry = self.job_is_entry[job_key]

        if is_entry:
            # 从 mobile 上传 input_size
            src_type, src_id = "MobileDevice", job.sourceDeviceId
            if (src_type, src_id) == (dst_type, dst_id):
                return 0.0
            bw = get_bandwidth_between(
                self.servers, self.devices,
                src_type, src_id, dst_type, dst_id)
            if bw <= 0 or bw == float('inf'):
                return 0.0
            trans_time = job.input_size / bw
            return trans_time

        # 非入口: 按源分组累加父任务 output_size
        source_groups = defaultdict(int)
        for pred_id in preds:
            pred_key = (wf_id, pred_id)
            pred_job = self.job_lookup[pred_key]
            data_source = self.job_output_location.get(
                pred_key, ("MobileDevice", job.sourceDeviceId))
            source_groups[data_source] += pred_job.output_size

        max_trans_time = 0.0
        for data_source, total_size in source_groups.items():
            if data_source == (dst_type, dst_id):
                continue
            bw = get_bandwidth_between(
                self.servers, self.devices,
                data_source[0], data_source[1], dst_type, dst_id)
            if bw <= 0 or bw == float('inf'):
                continue
            trans_time = total_size / bw
            if trans_time > max_trans_time:
                max_trans_time = trans_time

        return max_trans_time

    # ========================================================
    #  本地 SA
    # ========================================================
    def _try_process_local_job(self):
        with self.lock:
            if self.local_is_busy or not self.local_wait_list:
                return False
            if len(self.local_wait_list) == 1:
                idx = 0
            else:
                matrix = self._build_local_sa_matrix(self.local_wait_list)
                idx = self.decision.decide_sa(matrix)

            job_key = self.local_wait_list.pop(idx)
            self.local_is_busy = True
            self.local_processing = job_key

        self._execute_local_job(job_key)
        return True

    def _compute_sa_features(self, job_key, target_device):
        """返回归一化后的 8 个字段"""
        wf_id, jid = job_key
        job = self.job_lookup[job_key]
        wf = self.wf_by_id[wf_id]
        is_exit = self.job_is_exit[job_key]
        now = time.time() - self.start_time

        # 平均预算
        budget = max(1e-6, wf.deadline - wf.arrivalTime)
        avg_budget = budget / wf.num

        pt = job.workload / target_device.pa
        if is_exit and target_device.server_type != "MobileDevice":
            dt = job.output_size / target_device.bandwidth if target_device.bandwidth else 0.0
        else:
            dt = 0.0

        sn = len(self.job_children[job_key])
        max_children = max(1, max(len(self.job_children[(wf_id, j.id)])
                                  for j in wf.jobs))

        cr = wf.getCompleteRate()
        ur = job.UpwardRank or 0.0
        max_ur = max((j.UpwardRank or 0.0) for j in wf.jobs)
        if max_ur <= 0:
            max_ur = 1.0

        wt = now - (job.readyTime if job.readyTime is not None else now)
        mean_pa = sum(s.pa for s in self.servers) / len(self.servers)
        server_list_with_bw = [s for s in self.servers if s.bandwidth is not None]
        mean_bw = (sum(s.bandwidth for s in server_list_with_bw)
                   / len(server_list_with_bw)) if server_list_with_bw else 1.0
        ky = 1.0 if wf.is_on_critical_path(job, mean_pa, mean_bw) else 0.0
        dl = wf.deadline - now

        st1 = pt / avg_budget
        st2 = dt / avg_budget
        st3 = sn / max_children
        st4 = cr
        st5 = ur / max_ur
        st6 = wt / avg_budget
        st7 = ky
        st8 = dl / (dl + pt + dt + 1e-6)

        return [st1, st2, st3, st4, st5, st6, st7, st8]

    def _build_local_sa_matrix(self, job_keys):
        my_device = self.device_map[self.node_id]
        matrix = np.zeros((len(job_keys), 8), dtype=np.float64)
        for i, key in enumerate(job_keys):
            matrix[i] = self._compute_sa_features(key, my_device)
        return matrix

    def _execute_local_job(self, job_key):
        wf_id, jid = job_key
        job = self.job_lookup[job_key]
        my_device = self.device_map[self.node_id]

        pt = job.workload / my_device.pa
        is_exit = self.job_is_exit[job_key]
        dt = (job.output_size / my_device.bandwidth
              if is_exit and my_device.server_type != "MobileDevice"
              and my_device.bandwidth else 0.0)

        now = time.time() - self.start_time
        self.local_busy_until = now + pt + dt

        def _finish():
            self._on_local_job_finish(job_key)

        threading.Timer(pt + dt, _finish).start()

    def _on_local_job_finish(self, job_key):
        wf_id, jid = job_key
        job = self.job_lookup[job_key]
        is_exit = self.job_is_exit[job_key]
        now = time.time() - self.start_time

        with self.lock:
            job.processEndTime = now
            job.output_location = ("MobileDevice", self.device_id)
            self.job_output_location[job_key] = ("MobileDevice", self.device_id)
            self.job_process_end[job_key] = now

            if is_exit:
                job.finishTime = now
                job.state = "已完成"
            else:
                job.finishTime = now
                job.state = "执行完成"

            self.local_is_busy = False
            self.local_processing = None

            for child_id in self.job_children[job_key]:
                child_key = (wf_id, child_id)
                self.done_parents[child_key].add(jid)
                if (self.done_parents[child_key]
                        == set(self.job_parents[child_key])):
                    self.ready_set.add(child_key)
                    child_job = self.job_lookup.get(child_key)
                    if child_job is not None:
                        child_job.readyTime = now

        self._wakeup.set()

    # ========================================================
    #  /observe_sa 处理（返回归一化后的 8 维）
    # ========================================================
    def _handle_observe_sa(self, server_id, wf_id, job_ids):
        wf = self.wf_by_id.get(wf_id)
        if wf is None:
            return {"jobs": [[0.0] * 8 for _ in job_ids]}

        target_server = self.server_map.get(server_id)
        if target_server is None:
            return {"jobs": [[0.0] * 8 for _ in job_ids]}

        jobs_resp = []
        for jid in job_ids:
            key = (wf_id, jid)
            if key not in self.job_lookup:
                jobs_resp.append([0.0] * 8)
                continue
            jobs_resp.append(self._compute_sa_features(key, target_server))

        return {"jobs": jobs_resp}

    # ========================================================
    #  /notify_ready
    # ========================================================
    def _handle_notify_ready(self, body):
        wf_id = body["workflow_id"]
        completed_jid = body["completed_job_id"]
        output_loc = tuple(body["output_location"])
        process_end = body["process_end_time"]
        finish_time = body.get("finish_time")
        is_exit = body.get("is_exit", False)

        with self.lock:
            completed_key = (wf_id, completed_jid)
            self.job_output_location[completed_key] = output_loc
            self.job_process_end[completed_key] = process_end

            job = self.job_lookup.get(completed_key)
            if job is not None:
                job.processEndTime = process_end
                job.output_location = output_loc
                if is_exit and finish_time is not None:
                    job.finishTime = finish_time
                    job.state = "已完成"
                else:
                    job.finishTime = process_end
                    job.state = "执行完成"

            for child_id in self.job_children.get(completed_key, []):
                child_key = (wf_id, child_id)
                self.done_parents[child_key].add(completed_jid)
                if (self.done_parents[child_key]
                        == set(self.job_parents[child_key])):
                    self.ready_set.add(child_key)
                    child_job = self.job_lookup.get(child_key)
                    if child_job is not None:
                        child_job.readyTime = process_end

        self._wakeup.set()

    # ========================================================
    #  完成判断 + report
    # ========================================================
    def _all_workflows_done(self):
        with self.lock:
            for wf in self.my_workflows:
                for job in wf.jobs:
                    is_exit = (len(list(wf.graph.successors(job.id))) == 0)
                    if is_exit and job.finishTime is None:
                        return False
            return True

    def _report_done(self):
        finish_times = []
        makespan = 0.0
        for wf in self.my_workflows:
            wf_finish = wf.getWorkflowFinishTime()
            if wf_finish > makespan:
                makespan = wf_finish
            finish_times.append({
                "workflow_id": wf.id,
                "finish_time": wf_finish,
                "deadline": wf.deadline,
            })

        payload = make_report_done_req(
            node_id=self.node_id,
            finish_time=makespan,
            workflow_finish_times=finish_times,
        )
        # ---- 附上本节点的控制开销统计 ----
        payload["comm_stats"] = communication.get_stats()

        logger.info(f"[{self.node_id}] 完成, makespan={makespan:.2f}, "
                    f"workflows={len(self.my_workflows)}")
        communication.report_done(self.master_url, payload)