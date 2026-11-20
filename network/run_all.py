# network/run_all.py
"""
单机多进程一键启动

启动 1 个 Master + M 个 Mobile + E+C 个 Server
每个节点是一个独立的 multiprocessing.Process

用法:
    cd <项目根目录>
    python -m network.run_all
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import time
import logging
import multiprocessing as mp

# 让子进程能 import 项目根目录下的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network.config import (
    SCENARIO, NODES, MASTER_PORT, MASTER_URL,
    MOBILE_IDS, EDGE_IDS, CENTER_IDS,
)
from network import communication

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(processName)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ============================================================
#  子进程入口
# ============================================================
def run_master():
    from network.master import Master
    m = Master()
    m.run(host="0.0.0.0", port=MASTER_PORT)


def run_mobile(device_id):
    from network.mobile_node import MobileNode
    node = MobileNode(device_id, master_url=MASTER_URL)
    node_id = f"MobileDevice_{device_id}"
    port = _port_of(node_id)
    node.launch(host="0.0.0.0", port=port)


def run_server(server_id_int, server_type):
    from network.server_node import ServerNode
    node = ServerNode(server_id_int, server_type, master_url=MASTER_URL)
    node_id = f"{server_type}_{server_id_int}"
    port = _port_of(node_id)
    node.launch(host="0.0.0.0", port=port)


def _port_of(node_id):
    """从 URL 里解析端口"""
    url = NODES[node_id]         # "http://127.0.0.1:8001"
    return int(url.rsplit(":", 1)[-1])


# ============================================================
#  主进程
# ============================================================
def main():
    logger.info("=== 分布式测试启动 ===")
    logger.info(f"场景: {SCENARIO['wf_size']}-{SCENARIO['mobile_num']}, "
                f"wf_num={SCENARIO['wf_num']}")
    logger.info(f"节点: {len(MOBILE_IDS)} mobile + "
                f"{len(EDGE_IDS)} edge + {len(CENTER_IDS)} center")

    os.makedirs("logs", exist_ok=True)

    # 重置主进程统计（子进程是 spawn 的，各自独立，不受影响）
    communication.reset_stats()

    # 用 spawn 更干净（避免 fork 带来的状态继承问题）
    ctx = mp.get_context("spawn")
    processes = []

    # ---- 1. Master ----
    p_master = ctx.Process(target=run_master, name="Master")
    p_master.start()
    processes.append(p_master)
    logger.info("Master 已启动，等待 2 秒 ...")
    time.sleep(2.0)

    # ---- 2. Mobile ----
    for device_id in range(len(MOBILE_IDS)):
        p = ctx.Process(target=run_mobile, args=(device_id,),
                        name=f"Mobile_{device_id}")
        p.start()
        processes.append(p)

    # ---- 3. Edge Server ----
    for sid in range(len(EDGE_IDS)):
        p = ctx.Process(target=run_server, args=(sid, "EdgeServer"),
                        name=f"Edge_{sid}")
        p.start()
        processes.append(p)

    # ---- 4. Center Server ----
    for sid in range(len(CENTER_IDS)):
        p = ctx.Process(target=run_server, args=(sid, "CenterServer"),
                        name=f"Center_{sid}")
        p.start()
        processes.append(p)

    logger.info(f"所有 {len(processes)} 个进程已启动")

    # ---- 5. 等所有进程结束 ----
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C，终止所有进程 ...")
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(timeout=3)
        logger.info("已全部终止")


if __name__ == "__main__":
    main()