# network/node_base.py
"""
节点基类

职责:
  - 提供 Flask app 骨架
  - 提供公共路由: /health, /start, /shutdown
  - 提供注册、等待启动、优雅退出
  - 子类 (MobileNode / ServerNode) 继承后注册自己的业务路由

不做:
  - 业务逻辑（RA/SA 决策、job 管理）
  - 通信（那是 communication.py 的事）
"""
import time
import logging
import threading

from flask import Flask, request, jsonify

from network.config import MASTER_URL, RUNTIME
from network.protocol import make_health_resp, make_shutdown_resp
from network import communication

logger = logging.getLogger(__name__)


class BaseNode:
    def __init__(self, node_id, node_type, master_url=None):
        self.node_id = node_id
        self.node_type = node_type          # "MobileDevice" / "EdgeServer" / "CenterServer"
        self.master_url = master_url or MASTER_URL

        self.app = Flask(__name__)
        self.ready = False                  # 数据是否加载完
        self.busy = False                   # 是否在跑主循环
        self.started = threading.Event()    # 收到 /start 后 set
        self.shutdown_flag = threading.Event()
        self.start_time = None              # /start 时记录

        self.main_thread = None
        self._register_base_routes()

    # ========================================================
    #  公共路由
    # ========================================================
    def _register_base_routes(self):
        @self.app.route("/health", methods=["GET"])
        def _health():
            return jsonify(make_health_resp(
                node_id=self.node_id,
                node_type=self.node_type,
                ready=self.ready,
                busy=self.busy,
            ))

        @self.app.route("/start", methods=["POST"])
        def _start():
            if not self.ready:
                return jsonify({"status": "error",
                                "error": "node not ready"}), 400
            if self.started.is_set():
                return jsonify({"status": "ok",
                                "msg": "already started"})
            self.start_time = time.time()
            self.started.set()
            logger.info(f"[{self.node_id}] 收到 /start，开始主循环")
            return jsonify({"status": "ok"})

        @self.app.route("/shutdown", methods=["POST"])
        def _shutdown():
            logger.info(f"[{self.node_id}] 收到 /shutdown")
            self.shutdown_flag.set()
            return jsonify(make_shutdown_resp())

    # ========================================================
    #  注册到 Master
    # ========================================================
    def register_to_master(self):
        """向 Master 注册自己（带重试）"""
        payload = {
            "node_id": self.node_id,
            "node_type": self.node_type,
        }
        ok, data, err = communication.post(
            self.master_url, "/register", payload,
            timeout=5, retries=5, retry_delay=1.0)
        if ok:
            logger.info(f"[{self.node_id}] 注册成功: {data}")
            return True
        logger.error(f"[{self.node_id}] 注册失败: {err}")
        return False

    # ========================================================
    #  启动 HTTP 服务 + 主循环
    # ========================================================
    def start_http(self, host, port):
        """在后台线程启动 Flask（非阻塞）"""
        def _run():
            self.app.run(host=host, port=port, threaded=True,
                         use_reloader=False)
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        logger.info(f"[{self.node_id}] HTTP 服务启动在 {host}:{port}")

    def wait_for_start(self, timeout=None):
        """阻塞等待 Master 的 /start 信号"""
        self.started.wait(timeout=timeout)

    def run_main_loop(self):
        """
        子类必须重写：主循环
        - 内部应定期检查 self.shutdown_flag
        - 收到 shutdown_flag 后应尽快退出
        """
        raise NotImplementedError

    def launch(self, host, port):
        """
        完整启动流程:
          1. 启动 HTTP 服务
          2. 注册到 Master
          3. 等 /start
          4. 跑主循环
          5. 等 /shutdown 或主循环自然结束
        """
        self.start_http(host, port)

        if not self.register_to_master():
            logger.error(f"[{self.node_id}] 注册失败，退出")
            return

        # 先让子类做准备工作（加载数据/模型）
        self.prepare()

        logger.info(f"[{self.node_id}] 准备就绪，等待 /start")
        self.wait_for_start()

        if self.shutdown_flag.is_set():
            logger.info(f"[{self.node_id}] 启动前收到 shutdown，退出")
            return

        # 跑主循环
        logger.info(f"[{self.node_id}] 进入主循环")
        self.busy = True
        try:
            self.run_main_loop()
        except Exception as e:
            logger.exception(f"[{self.node_id}] 主循环异常")
        finally:
            self.busy = False
            logger.info(f"[{self.node_id}] 主循环结束")

        # 等 shutdown（如果还没收到）
        if not self.shutdown_flag.is_set():
            logger.info(f"[{self.node_id}] 主循环结束，等待 /shutdown")
            self.shutdown_flag.wait(timeout=60)

        logger.info(f"[{self.node_id}] 退出")

    def prepare(self):
        """
        子类可重写：加载数据、模型等
        在注册成功后、等 /start 前调用
        """
        self.ready = True