# network/communication.py
"""
HTTP 通信封装

设计:
  - 同步请求，带超时和重试
  - fan_out: 向多个 URL 并发发同一请求，等所有回复
  - 所有方法返回 (ok: bool, data: dict | None, error: str | None)
  - 全局统计控制开销（请求数、收发字节数）
"""
import json
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from network.config import COMM

logger = logging.getLogger(__name__)


# ============================================================
#  控制开销统计（全局，线程安全）
# ============================================================
STATS = {
    "requests": 0,          # 总请求次数（含重试）
    "bytes_tx": 0,          # 发送 JSON 字节数
    "bytes_rx": 0,          # 接收 JSON 字节数
    "by_endpoint": {},      # endpoint -> 请求次数
}
_stats_lock = threading.Lock()


def _payload_size(obj):
    try:
        return len(json.dumps(obj, separators=(",", ":")).encode("utf-8"))
    except Exception:
        return 0


def _record(endpoint, tx, rx):
    with _stats_lock:
        STATS["requests"] += 1
        STATS["bytes_tx"] += tx
        STATS["bytes_rx"] += rx
        STATS["by_endpoint"][endpoint] = STATS["by_endpoint"].get(endpoint, 0) + 1


def reset_stats():
    with _stats_lock:
        STATS["requests"] = 0
        STATS["bytes_tx"] = 0
        STATS["bytes_rx"] = 0
        STATS["by_endpoint"] = {}


def get_stats():
    with _stats_lock:
        return {
            "requests": STATS["requests"],
            "bytes_tx": STATS["bytes_tx"],
            "bytes_rx": STATS["bytes_rx"],
            "by_endpoint": dict(STATS["by_endpoint"]),
        }


# ============================================================
#  单次请求（带重试）
# ============================================================
def _post_once(url, endpoint, payload, timeout):
    full_url = url.rstrip("/") + endpoint
    tx = _payload_size(payload)
    try:
        r = requests.post(full_url, json=payload, timeout=timeout)
        if r.status_code != 200:
            _record(endpoint, tx, 0)
            return False, None, f"HTTP {r.status_code}: {r.text[:200]}"
        try:
            data = r.json()
        except Exception:
            data = None
        rx = _payload_size(data) if data is not None else len(r.content or b"")
        _record(endpoint, tx, rx)
        return True, data, None
    except requests.exceptions.Timeout:
        _record(endpoint, tx, 0)
        return False, None, "timeout"
    except requests.exceptions.ConnectionError as e:
        _record(endpoint, tx, 0)
        return False, None, f"connection_error: {e}"
    except Exception as e:
        _record(endpoint, tx, 0)
        return False, None, f"unknown: {e}"


def post(url, endpoint, payload, timeout=None, retries=None, retry_delay=None):
    """
    同步 POST，带重试
    返回 (ok, data, error)
    """
    if timeout is None:
        timeout = COMM["timeout_submit"]
    if retries is None:
        retries = COMM["max_retries"]
    if retry_delay is None:
        retry_delay = COMM["retry_delay"]

    last_err = None
    for attempt in range(retries + 1):
        ok, data, err = _post_once(url, endpoint, payload, timeout)
        if ok:
            return True, data, None
        last_err = err
        if attempt < retries:
            time.sleep(retry_delay)
    return False, None, last_err


# ============================================================
#  fan-out：向多个 URL 并发发同一请求，等所有回复
# ============================================================
def fan_out(urls, endpoint, payload, timeout=None, max_workers=None):
    """
    向多个 URL 并发发同一请求
    返回 list[(url, ok, data, error)]，顺序与 urls 一致
    """
    if timeout is None:
        timeout = COMM["timeout_observe_ra"]
    if max_workers is None:
        max_workers = min(COMM["max_workers"], max(1, len(urls)))

    results = [None] * len(urls)

    def _call(idx, url):
        ok, data, err = post(url, endpoint, payload, timeout=timeout,
                             retries=COMM["max_retries"])
        results[idx] = (url, ok, data, err)

    if len(urls) == 0:
        return []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_call, i, u) for i, u in enumerate(urls)]
        for f in as_completed(futures):
            f.result()  # 异常已被 _call 内部吞掉

    return results


# ============================================================
#  便捷封装（针对不同接口，带默认超时）
# ============================================================
def observe_ra(url, payload):
    return post(url, "/observe_ra", payload,
                timeout=COMM["timeout_observe_ra"])


def observe_ra_fan_out(urls, payload):
    return fan_out(urls, "/observe_ra", payload,
                   timeout=COMM["timeout_observe_ra"])


def observe_sa(url, payload):
    return post(url, "/observe_sa", payload,
                timeout=COMM["timeout_observe_sa"])


def observe_sa_fan_out(urls, payload):
    return fan_out(urls, "/observe_sa", payload,
                   timeout=COMM["timeout_observe_sa"])


def reserve(url, payload):
    return post(url, "/reserve", payload,
                timeout=COMM["timeout_submit"])


def submit_job(url, payload):
    return post(url, "/submit_job", payload,
                timeout=COMM["timeout_submit"])


def notify_ready(url, payload):
    return post(url, "/notify_ready", payload,
                timeout=COMM["timeout_notify"])


def report_done(url, payload):
    return post(url, "/report_done", payload,
                timeout=COMM["timeout_report"])


def shutdown(url):
    return post(url, "/shutdown", {}, timeout=COMM["timeout_report"])


def health(url):
    full_url = url.rstrip("/") + "/health"
    try:
        r = requests.get(full_url, timeout=5)
        if r.status_code == 200:
            return True, r.json(), None
        return False, None, f"HTTP {r.status_code}"
    except Exception as e:
        return False, None, str(e)