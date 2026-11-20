# network/config.py
"""
分布式测试系统的全局配置

- NODES:           节点 ID -> URL 映射（每个节点一个 Flask 进程）
- SCENARIO:        场景参数（节点数量、workflow 数量等）
- ALGO:            本次实验用哪个决策算法
- MODEL_PATHS:     RA/SA 权重路径（各节点本地加载）
- 其他超时/并发参数
"""
import os

# ============================================================
#  项目根目录（用于定位模型文件）
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
#  节点地址表
#  规则:
#    - MobileDevice_i   占用 8001 ~
#    - EdgeServer_i     接着排
#    - CenterServer_i   接着排
#  Master 单独用 8100
# ============================================================
def build_nodes(mobile_num, edge_num, center_num, host="127.0.0.1", base_port=8001):
    """
    生成节点 ID -> URL 的映射
    返回 (nodes_dict, mobile_ids, edge_ids, center_ids)
    """
    nodes = {}
    mobile_ids, edge_ids, center_ids = [], [], []

    port = base_port
    for i in range(mobile_num):
        nid = f"MobileDevice_{i}"
        nodes[nid] = f"http://{host}:{port}"
        mobile_ids.append(nid)
        port += 1
    for i in range(edge_num):
        nid = f"EdgeServer_{i}"
        nodes[nid] = f"http://{host}:{port}"
        edge_ids.append(nid)
        port += 1
    for i in range(center_num):
        nid = f"CenterServer_{i}"
        nodes[nid] = f"http://{host}:{port}"
        center_ids.append(nid)
        port += 1

    return nodes, mobile_ids, edge_ids, center_ids


# ============================================================
#  场景参数
# ============================================================
SCENARIO = {
    # 节点数量
    "mobile_num": 3,
    "edge_num":   4,
    "center_num": 4,

    # workflow
    "wf_num":      5,
    "wf_size":     "small",   # small / medium / large
    "lambda_rate": 0.1,

    # 设备资源参数（与 env.lib.createDataset 保持一致）
    "mobile_pa_range":   (750, 1500),
    "edge_pa_range":     (1500, 3000),
    "edge_bandwidth":    1024 * 1024 * 1024,        # 1024 Mb/s
    "center_pa_range":   (3000, 6000),
    "center_bandwidth_choices": ([128, 256, 512], [0.2, 0.6, 0.2]),

    # 随机种子（保证可复现）
    "seed": 52,
}

MASTER_PORT = 8100
MASTER_URL  = f"http://127.0.0.1:{MASTER_PORT}"

# 生成节点表
NODES, MOBILE_IDS, EDGE_IDS, CENTER_IDS = build_nodes(
    SCENARIO["mobile_num"], SCENARIO["edge_num"], SCENARIO["center_num"]
)

# 节点 ID 列表（排序稳定，便于索引）
ALL_NODE_IDS = MOBILE_IDS + EDGE_IDS + CENTER_IDS

# server 节点（用于 RA 候选；不含 mobile）
SERVER_IDS = EDGE_IDS + CENTER_IDS


# ============================================================
#  算法选择
# ============================================================
# 可选: "DERL" / "HEFT" / "MTGP" / "ULS"
ALGO = "DERL"


# ============================================================
#  模型路径（各节点本地加载；路径相对于项目根目录）
# ============================================================
BASE_PATH = "RS20260926173842331751"
EPISODE = 270
MODEL_PATHS = {
    "ra_path1": f"logs/{BASE_PATH}/saved_models/ep_{EPISODE}_r.pt",
    "ra_path2": f"logs/{BASE_PATH}/saved_models/ob_rms_{EPISODE}_r.pickle",
    "sa_path1": f"logs/{BASE_PATH}/saved_models/ep_{EPISODE}_s.pt",
    "sa_path2": f"logs/{BASE_PATH}/saved_models/ob_rms_{EPISODE}_s.pickle",
    "mtgp_path": "compares/gp/logs/GP20251207115911656482/saved_models/gen_49_best_ind.pkl",
}


# ============================================================
#  通信参数
# ============================================================
COMM = {
    "timeout_observe_ra": 30,   # seconds
    "timeout_observe_sa": 30,
    "timeout_submit":     30,
    "timeout_notify":     30,
    "timeout_report":     10,
    "max_workers":        32,    # fan-out 线程池大小
    "max_retries":        2,
    "retry_delay":        0.5,
}


# ============================================================
#  运行时参数
# ============================================================
RUNTIME = {
    "poll_interval": 0.005,       # 主循环轮询间隔（秒）
    "main_loop_sleep": 0.001,     # 忙等时 sleep
    "shutdown_grace": 2.0,        # 收到 shutdown 后等待多久退出
}


def resolve_path(rel_path: str) -> str:
    """把 config 里的相对路径转成绝对路径"""
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.join(PROJECT_ROOT, rel_path)