# network/server_node.py
"""
Server 节点（Edge / Center）—— 新版模型
"""
import os
import time
import logging
import threading

import numpy as np
from flask import request, jsonify

from network.config import SCENARIO, NODES, RUNTIME
from network.node_base import BaseNode
from network.reuse import (
    build_resources, get_bandwidth_between, job_from_snapshot,
)
from network import communication
from network.protocol import (
    make_observe_ra_resp,
    make_reserve_resp, make_submit_resp,
    make_notify_ready_req,
)
from network.decision import get_decision

logger = logging.getLogger(__name__)


class ServerNode(BaseNode):
    def __init__(self, server_id_int, server_type, master_url=None):
        node_id = f"{server_type}_{server_id_int}"
        super().__init__(node_id, server_type, master_url)
        self.server_id = server_id_int
        self.server_type = server_type

        self.devices = None
        self.servers = None
        self.server_map = {}
        self.device_map = {}
        self.url_of = {}
        self.my_server = None

        self.wait_list = []
        self.incoming_list = []
        self.processing = None
        self.is_busy = False
        self.busy_until = 0.0

        self.lock = threading.RLock()
        self._wakeup = threading.Event()
        self.decision = None

        self._register_business_routes()

    def _register_business_routes(self):
        @self.app.route("/observe_ra", methods=["POST"])
        def _observe_ra():
            body = request.get_json(force=True)
            return jsonify(self._handle_observe_ra(body))

        @self.app.route("/reserve", methods=["POST"])
        def _reserve():
            body = request.get_json(force=True)
            return jsonify(self._handle_reserve(body))

        @self.app.route("/submit_job", methods=["POST"])
        def _submit_job():
            body = request.get_json(force=True)
            return jsonify(self._handle_submit_job(body))

    def prepare(self):
        logger.info(f"[{self.node_id}] 加载资源 ...")
        seed = SCENARIO["seed"]

        devices, servers = build_resources(seed=seed)
        self.devices = devices
        self.servers = servers

        for d in devices:
            self.device_map[d.name] = d
        for s in servers:
            self.server_map[s.name] = s

        self.url_of = dict(NODES)
        self.my_server = self.server_map[self.node_id]

        self.decision = get_decision()
        self.ready = True
        logger.info(f"[{self.node_id}] prepare 完成, pa={self.my_server.pa:.1f}, "
                    f"bw={self.my_server.bandwidth/1024/1024:.0f} Mb/s")

    def run_main_loop(self):
        while not self.shutdown_flag.is_set():
            try:
                self._try_process_one_job()
            except Exception:
                logger.exception(f"[{self.node_id}] 主循环异常")
            self._wakeup.wait(timeout=RUNTIME["main_loop_sleep"])
            self._wakeup.clear()

    def _try_process_one_job(self):
        with self.lock:
            if self.is_busy or not self.wait_list:
                return
            if len(self.wait_list) == 1:
                idx = 0
            else:
                matrix = self._build_sa_matrix(self.wait_list)
                idx = self.decision.decide_sa(matrix)

            job_entry = self.wait_list.pop(idx)
            self.is_busy = True
            self.processing = job_entry

        self._execute_job(job_entry)

    def _build_sa_matrix(self, wait_list):
        """
        SA 矩阵: [pt, dt, sn, cr, ur, wt, ky, dl]
        - mobile 端返回归一化后的 8 维
        - server 端预填占位（pt/dt 本地算，其他为 0），mobile 返回后覆盖
        """
        matrix = np.zeros((len(wait_list), 8), dtype=np.float64)
        for i, entry in enumerate(wait_list):
            pt = entry["snapshot"]["workload"] / self.my_server.pa
            is_exit = entry["is_exit"]
            dt = (entry["snapshot"]["output_size"] / self.my_server.bandwidth
                  ) if is_exit and self.my_server.bandwidth else 0.0
            matrix[i] = [pt, dt, 0, 0, 0, 0, 0, 0]

        # 按 source_device + wf_id 分组
        groups = {}
        for i, entry in enumerate(wait_list):
            key = (entry["source_device"], entry["wf_id"])
            groups.setdefault(key, []).append((i, entry))

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _ask(key, items):
            src, wf_id = key
            url = self.url_of.get(src)
            if url is None:
                return []
            job_ids = [e["jid"] for _, e in items]
            payload = {
                "server_id": self.node_id,
                "workflow_id": wf_id,
                "job_ids": job_ids,
            }
            ok, data, err = communication.post(
                url, "/observe_sa", payload, timeout=30, retries=2)
            if not ok:
                logger.warning(f"[{self.node_id}] observe_sa to {src} 失败: {err}")
                return []
            out = []
            for k, (idx, _) in enumerate(items):
                out.append((idx, data["jobs"][k]))
            return out

        all_results = []
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(groups)))) as ex:
            futures = [ex.submit(_ask, key, items) for key, items in groups.items()]
            for f in as_completed(futures):
                all_results.extend(f.result())

        # 覆盖为 mobile 返回的归一化值
        for idx, vec in all_results:
            matrix[idx] = vec

        return matrix

    def _handle_observe_ra(self, body):
        """
        返回原始值（归一化在 mobile 端做）
        支持通过环境变量 OBSERVE_RA_DELAY_MS 注入响应延迟，
        用于 stale information 消融实验。
        """
        # ---- stale 消融：人为延迟响应 ----
        _delay_ms = int(os.environ.get("OBSERVE_RA_DELAY_MS", "0"))
        if _delay_ms > 0:
            time.sleep(_delay_ms / 1000.0)

        now = time.time() - self.start_time
        ut = body["ut"]

        with self.lock:
            ft = max(0.0, self.busy_until - now) if self.is_busy else 0.0
            arrival_deadline = now + ut

            wn = len(self.wait_list)
            wp = sum(e["snapshot"]["workload"] / self.my_server.pa
                     for e in self.wait_list)

            for inc in self.incoming_list:
                if inc["arrive_time"] < arrival_deadline:
                    wn += 1
                    wp += inc["workload"] / self.my_server.pa

            server_free = max(now, self.busy_until) if self.is_busy else now
            wd = max(arrival_deadline, server_free) - now

            return make_observe_ra_resp(
                ft=ft, wn=wn, wp=wp, wd=wd
            )

    def _handle_reserve(self, body):
        wf_id = body["workflow_id"]
        jid = body["job_id"]
        ut = body["ut"]
        src = body["source_device"]
        workload = body["workload"]
        now = time.time() - self.start_time

        with self.lock:
            self.incoming_list.append({
                "wf_id": wf_id,
                "jid": jid,
                "arrive_time": now + ut,
                "source_device": src,
                "workload": workload,
            })
            queue_len = len(self.wait_list) + len(self.incoming_list)

        return make_reserve_resp(status="ok", queue_len=queue_len)

    def _handle_submit_job(self, body):
        wf_id = body["workflow_id"]
        jid = body["job_id"]
        snapshot = body["job_snapshot"]
        src = body["source_device"]
        is_exit = body["is_exit"]

        with self.lock:
            self.incoming_list = [
                inc for inc in self.incoming_list
                if not (inc["wf_id"] == wf_id and inc["jid"] == jid)
            ]
            job_obj = job_from_snapshot(jid, wf_id, snapshot)
            self.wait_list.append({
                "wf_id": wf_id,
                "jid": jid,
                "snapshot": snapshot,
                "source_device": src,
                "is_exit": is_exit,
                "job_obj": job_obj,
            })
        self._wakeup.set()
        return make_submit_resp(status="ok")

    def _execute_job(self, job_entry):
        snapshot = job_entry["snapshot"]
        is_exit = job_entry["is_exit"]
        pt = snapshot["workload"] / self.my_server.pa
        dt = (snapshot["output_size"] / self.my_server.bandwidth
              ) if is_exit and self.my_server.bandwidth else 0.0

        now = time.time() - self.start_time
        self.busy_until = now + pt + dt

        def _finish():
            self._on_job_finish(job_entry)
        threading.Timer(pt + dt, _finish).start()

    def _on_job_finish(self, job_entry):
        wf_id = job_entry["wf_id"]
        jid = job_entry["jid"]
        is_exit = job_entry["is_exit"]
        src = job_entry["source_device"]
        now = time.time() - self.start_time

        with self.lock:
            self.is_busy = False
            self.processing = None

        url = self.url_of.get(src)
        if url is not None:
            payload = make_notify_ready_req(
                workflow_id=wf_id,
                completed_job_id=jid,
                output_location=[self.server_type, self.server_id],
                process_end_time=now,
                finish_time=now if is_exit else None,
                is_exit=is_exit,
            )
            communication.notify_ready(url, payload)
        self._wakeup.set()