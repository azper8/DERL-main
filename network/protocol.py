# network/protocol.py
"""
消息构造与解析
"""
def job_key(workflow_id, job_id):
    return (int(workflow_id), int(job_id))


def make_register_req(node_id, node_type):
    return {"node_id": node_id, "node_type": node_type}


def make_observe_ra_req(workflow_id, job_id, ut, pt, dt, dl):
    return {
        "workflow_id": int(workflow_id),
        "job_id":      int(job_id),
        "ut": float(ut),
        "pt": float(pt),
        "dt": float(dt),
        "dl": float(dl)
    }


def make_observe_ra_resp(ft, wn, wp, wd):
    return {
        "ft":   float(ft),
        "wn":   int(wn),
        "wp":   float(wp),
        "wd":   float(wd)
    }


def make_observe_sa_req(server_id, workflow_id, job_ids):
    return {
        "server_id":   str(server_id),
        "workflow_id": int(workflow_id),
        "job_ids":     [int(x) for x in job_ids],
    }


def make_observe_sa_resp(jobs):
    return {"jobs": jobs}


def make_reserve_req(workflow_id, job_id, ut, source_device, workload):
    return {
        "workflow_id":   int(workflow_id),
        "job_id":        int(job_id),
        "ut":            float(ut),
        "source_device": str(source_device),
        "workload":      float(workload),
    }


def make_reserve_resp(status="ok", queue_len=0):
    return {"status": status, "queue_len": int(queue_len)}


def make_submit_req(workflow_id, job_id, job_snapshot, source_device, is_exit):
    """
    job_snapshot: {
        workload, input_size, output_size, source_device
    }
    """
    return {
        "workflow_id":   int(workflow_id),
        "job_id":        int(job_id),
        "job_snapshot":  job_snapshot,
        "source_device": str(source_device),
        "is_exit":       bool(is_exit),
    }


def make_submit_resp(status="ok"):
    return {"status": status}


def make_notify_ready_req(workflow_id, completed_job_id,
                          output_location, process_end_time,
                          finish_time=None, is_exit=False):
    return {
        "workflow_id":      int(workflow_id),
        "completed_job_id": int(completed_job_id),
        "output_location":  list(output_location),
        "process_end_time": float(process_end_time),
        "finish_time":      float(finish_time) if finish_time is not None else None,
        "is_exit":          bool(is_exit),
    }


def make_notify_ready_resp(status="ok"):
    return {"status": status}


def make_report_done_req(node_id, finish_time, workflow_finish_times):
    return {
        "node_id":               str(node_id),
        "finish_time":           float(finish_time),
        "workflow_finish_times": list(workflow_finish_times),
    }


def make_report_done_resp(status="ok", all_done=False):
    return {"status": status, "all_done": bool(all_done)}


def make_shutdown_req():
    return {}


def make_shutdown_resp(status="ok"):
    return {"status": status}


def make_health_resp(node_id, node_type, ready, busy):
    return {
        "node_id":   str(node_id),
        "node_type": str(node_type),
        "ready":     bool(ready),
        "busy":      bool(busy),
    }