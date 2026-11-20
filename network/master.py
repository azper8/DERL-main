# network/master.py
"""
Master（协调者）

职责:
  1. 接收各节点注册，知道有哪些节点
  2. 等所有节点 ready 后，广播 /start
  3. 收集各 mobile 的 /report_done
  4. 所有 mobile 完成后，计算 makespan / violation
  5. 广播 /shutdown
  6. 写结果 CSV（含控制开销统计）

不做:
  - 不参与调度决策
  - 不持有 workflow / job
"""
import os
import csv
import time
import logging
import threading

import numpy as np
from flask import Flask, request, jsonify

from network.config import (
    SCENARIO, MASTER_PORT, NODES, MOBILE_IDS,
    ALL_NODE_IDS, RUNTIME,
)
from network import communication

logger = logging.getLogger(__name__)

from pathlib import Path
_DEFAULT_CSV = str(Path(__file__).resolve().parent / "logs" / "output_distributed.csv")


class Master:
    def __init__(self, output_csv=_DEFAULT_CSV):
        self.app = Flask(__name__)
        self.output_csv = output_csv

        # 状态
        self.registered = {}          # node_id -> {node_type, url}
        self.done_reports = {}        # node_id -> report payload
        self.start_time = None
        self.lock = threading.Lock()
        self.started = False

        # 期望节点数（从 config 读）
        self.expected_node_num = len(ALL_NODE_IDS)
        self.expected_mobile_num = len(MOBILE_IDS)

        self._register_routes()

    # ========================================================
    #  路由
    # ========================================================
    def _register_routes(self):
        @self.app.route("/register", methods=["POST"])
        def _register():
            body = request.get_json(force=True)
            node_id = body["node_id"]
            node_type = body["node_type"]
            url = NODES.get(node_id)

            with self.lock:
                if node_id not in self.registered:
                    self.registered[node_id] = {
                        "node_type": node_type,
                        "url": url,
                    }
                    logger.info(f"节点注册: {node_id} ({node_type}), "
                                f"当前 {len(self.registered)}/{self.expected_node_num}")

                all_ready = len(self.registered) >= self.expected_node_num

            if all_ready and not self.started:
                # 触发开始
                threading.Thread(target=self._broadcast_start,
                                 daemon=True).start()

            return jsonify({"status": "ok",
                            "count": len(self.registered)})

        @self.app.route("/report_done", methods=["POST"])
        def _report_done():
            body = request.get_json(force=True)
            node_id = body["node_id"]
            with self.lock:
                self.done_reports[node_id] = body
                all_done = len(self.done_reports) >= self.expected_mobile_num

            logger.info(f"收到 {node_id} 完成报告, "
                        f"当前 {len(self.done_reports)}/{self.expected_mobile_num}")

            if all_done:
                threading.Thread(target=self._finalize, daemon=True).start()

            return jsonify({"status": "ok", "all_done": all_done})

        @self.app.route("/health", methods=["GET"])
        def _health():
            return jsonify({
                "status": "alive",
                "registered": len(self.registered),
                "expected": self.expected_node_num,
                "done": len(self.done_reports),
                "started": self.started,
            })

        @self.app.route("/start", methods=["POST"])
        def _manual_start():
            """手动触发（调试用）"""
            threading.Thread(target=self._broadcast_start,
                             daemon=True).start()
            return jsonify({"status": "ok"})

    # ========================================================
    #  广播 /start
    # ========================================================
    def _broadcast_start(self):
        if self.started:
            return
        self.started = True
        self.start_time = time.time()

        # 按类型分组
        server_nodes = [(nid, info) for nid, info in self.registered.items()
                        if info["node_type"] in ("EdgeServer", "CenterServer")]
        mobile_nodes = [(nid, info) for nid, info in self.registered.items()
                        if info["node_type"] == "MobileDevice"]

        # ---- 阶段 1：启动所有 server，串行同步等确认 ----
        logger.info(f"=== 阶段 1: 广播 /start 给 {len(server_nodes)} 个 server ===")
        for node_id, info in server_nodes:
            ok = False
            last_err = None
            for attempt in range(3):
                ok, data, err = communication.post(
                    info["url"], "/start", {}, timeout=10, retries=2)
                if ok:
                    logger.info(f"  {node_id} 已启动")
                    break
                last_err = err
                logger.warning(f"  {node_id} 第 {attempt + 1} 次启动失败: {err}")
                time.sleep(0.5)
            if not ok:
                logger.error(f"  {node_id} 启动最终失败: {last_err}")

        # ---- 阶段 2：所有 server 确认后才启动 mobile ----
        logger.info(f"=== 阶段 2: 广播 /start 给 {len(mobile_nodes)} 个 mobile ===")
        for node_id, info in mobile_nodes:
            ok = False
            last_err = None
            for attempt in range(3):
                ok, data, err = communication.post(
                    info["url"], "/start", {}, timeout=10, retries=2)
                if ok:
                    logger.info(f"  {node_id} 已启动")
                    break
                last_err = err
                logger.warning(f"  {node_id} 第 {attempt + 1} 次启动失败: {err}")
                time.sleep(0.5)
            if not ok:
                logger.error(f"  {node_id} 启动最终失败: {last_err}")

    # ========================================================
    #  汇总 + shutdown
    # ========================================================
    def _finalize(self):
        time.sleep(0.5)  # 等最后几个 report 稳定
        logger.info("=== 所有 mobile 完成，开始汇总 ===")

        # ---- 收集所有 workflow 的 finish/deadline ----
        all_finish_times = []
        all_violations = 0
        global_makespan = 0.0

        for node_id, rep in self.done_reports.items():
            ft = rep.get("finish_time", 0.0)
            if ft > global_makespan:
                global_makespan = ft

            for wf_info in rep.get("workflow_finish_times", []):
                f = wf_info.get("finish_time", 0.0)
                d = wf_info.get("deadline", 0.0)
                all_finish_times.append(f)
                if f > d:
                    all_violations += 1

        total_wf = len(all_finish_times)
        violation_rate = (all_violations / total_wf * 100) if total_wf > 0 else 0.0

        logger.info(f"全局 makespan = {global_makespan:.2f}")
        logger.info(f"全局 violation rate = {violation_rate:.2f}% "
                    f"({all_violations}/{total_wf})")

        # ---- 控制开销：累加各 mobile 上报 ----
        ctrl_req = 0
        ctrl_tx = 0
        ctrl_rx = 0
        by_endpoint_total = {}
        for node_id, rep in self.done_reports.items():
            cs = rep.get("comm_stats") or {}
            ctrl_req += int(cs.get("requests", 0))
            ctrl_tx += int(cs.get("bytes_tx", 0))
            ctrl_rx += int(cs.get("bytes_rx", 0))
            for ep, cnt in (cs.get("by_endpoint") or {}).items():
                by_endpoint_total[ep] = by_endpoint_total.get(ep, 0) + int(cnt)

        logger.info(f"控制开销(仅 mobile 侧发起): requests={ctrl_req}, "
                    f"tx={ctrl_tx}B, rx={ctrl_rx}B")
        logger.info(f"按接口: {by_endpoint_total}")

        # ---- 写 CSV ----
        os.makedirs(os.path.dirname(self.output_csv) or ".", exist_ok=True)
        write_header = not os.path.exists(self.output_csv)
        with open(self.output_csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["Scenario", "Makespan", "ViolationRate",
                            "TotalWorkflows", "Violations",
                            "CtrlReq", "CtrlBytesTx", "CtrlBytesRx"])
            scenario = f"{SCENARIO['wf_size']}-{SCENARIO['mobile_num']}"
            w.writerow([scenario,
                        f"{global_makespan:.2f}",
                        f"{violation_rate:.2f}",
                        total_wf, all_violations,
                        ctrl_req, ctrl_tx, ctrl_rx])

        logger.info(f"结果写入 {self.output_csv}")

        # ---- 广播 shutdown ----
        logger.info("=== 广播 /shutdown ===")
        for node_id, info in self.registered.items():
            url = info["url"]
            communication.shutdown(url)

    # ========================================================
    #  启动
    # ========================================================
    def run(self, host="0.0.0.0", port=None):
        port = port or MASTER_PORT
        logger.info(f"Master 启动在 {host}:{port}")
        logger.info(f"期望节点数: {self.expected_node_num}")
        logger.info(f"期望 mobile 数: {self.expected_mobile_num}")
        self.app.run(host=host, port=port, threaded=True, use_reloader=False)