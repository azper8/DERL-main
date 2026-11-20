# network/decision.py
"""
决策层 —— 当前只实现 DERL

设计:
  - observe 层负责通过 RPC 聚合，拼成状态矩阵
  - 本文件只负责：拿状态矩阵 → 输出 action 索引
  - 所有算法都实现同一个接口: decide_ra(matrix) / decide_sa(matrix)
  - 未来加 HEFT/MTGP/ULS/DRL-EA 时，只需扩展对应的类

DERL 的字段约定:
  RA 矩阵: (n_candidates, 8)  [ft, wn, wp, wd, ut, pt, dt, dl]
  SA 矩阵: (n_jobs, 8)        [pt, dt, sn, cr, ur, wt, ky, dl]
"""
import pickle

import numpy as np
import torch

from network.config import ALGO, MODEL_PATHS, resolve_path
from builder import build_policy


# ============================================================
#  字段索引（DERL）
# ============================================================
RA_FIELDS = ["ft", "wn", "wp", "wd", "ut", "pt", "dt", "dl"]
RA_IDX = {name: i for i, name in enumerate(RA_FIELDS)}

SA_FIELDS = ["pt", "dt", "sn", "cr", "ur", "wt", "ky", "dl"]
SA_IDX = {name: i for i, name in enumerate(SA_FIELDS)}


# ============================================================
#  DERL 决策器
# ============================================================
class DERLDecision:
    """DERL: RA/SA 都用 RL policy"""

    def __init__(self):
        policy_cfg = {
            'name': 'rl',
            'state_num': 8,
            'action_num': 1,
            'discrete_action': True,
            'add_gru': False,
            'action_type': 'greedy',
        }

        # ---- RA policy ----
        self.ra_policy = build_policy(policy_cfg)
        self.ra_policy.load_state_dict(
            torch.load(resolve_path(MODEL_PATHS["ra_path1"]),
                       map_location='cpu'))
        self.ra_policy.eval()

        # ---- SA policy ----
        self.sa_policy = build_policy(policy_cfg)
        self.sa_policy.load_state_dict(
            torch.load(resolve_path(MODEL_PATHS["sa_path1"]),
                       map_location='cpu'))
        self.sa_policy.eval()

        # ---- 观测归一化参数 ----
        with open(resolve_path(MODEL_PATHS["ra_path2"]), "rb") as f:
            ra_rms = pickle.load(f)
            half = len(ra_rms) // 2
            self.ra_mean = np.asarray(ra_rms[:half], dtype=np.float64)
            self.ra_std = np.asarray(ra_rms[half:], dtype=np.float64)

        with open(resolve_path(MODEL_PATHS["sa_path2"]), "rb") as f:
            sa_rms = pickle.load(f)
            half = len(sa_rms) // 2
            self.sa_mean = np.asarray(sa_rms[:half], dtype=np.float64)
            self.sa_std = np.asarray(sa_rms[half:], dtype=np.float64)

    # --------------------------------------------------------
    #  RA: 输入 (n_candidates, 8)，输出 action 索引
    # --------------------------------------------------------
    def decide_ra(self, state_matrix: np.ndarray) -> int:
        assert state_matrix.ndim == 2 and state_matrix.shape[1] == 8, \
            f"RA state shape 应为 (n, 8)，实际 {state_matrix.shape}"

        norm = (state_matrix - self.ra_mean) / self.ra_std
        action = self.ra_policy(norm, removed=[])
        return int(action)

    # --------------------------------------------------------
    #  SA: 输入 (n_jobs, 8)，输出 action 索引
    # --------------------------------------------------------
    def decide_sa(self, state_matrix: np.ndarray) -> int:
        assert state_matrix.ndim == 2 and state_matrix.shape[1] == 8, \
            f"SA state shape 应为 (n, 8)，实际 {state_matrix.shape}"

        norm = (state_matrix - self.sa_mean) / self.sa_std
        action = self.sa_policy(norm, removed=[])
        return int(action)


# ============================================================
#  占位类（未来扩展）
# ============================================================
class HEFTPDecision:
    def decide_ra(self, state_matrix): raise NotImplementedError
    def decide_sa(self, state_matrix): raise NotImplementedError


class MTGPDecision:
    def decide_ra(self, state_matrix): raise NotImplementedError
    def decide_sa(self, state_matrix): raise NotImplementedError


class ULSDecision:
    def decide_ra(self, state_matrix): raise NotImplementedError
    def decide_sa(self, state_matrix): raise NotImplementedError


class DRLEADecision:
    def decide_ra(self, state_matrix): raise NotImplementedError
    def decide_sa(self, state_matrix): raise NotImplementedError


# ============================================================
#  工厂（单例）
# ============================================================
_decision_singleton = None


def get_decision():
    global _decision_singleton
    if _decision_singleton is not None:
        return _decision_singleton

    if ALGO == "DERL":
        _decision_singleton = DERLDecision()
    elif ALGO == "HEFT":
        _decision_singleton = HEFTPDecision()
    elif ALGO == "MTGP":
        _decision_singleton = MTGPDecision()
    elif ALGO == "ULS":
        _decision_singleton = ULSDecision()
    elif ALGO == "DRL-EA":
        _decision_singleton = DRLEADecision()
    else:
        raise ValueError(f"未知算法: {ALGO}")

    return _decision_singleton