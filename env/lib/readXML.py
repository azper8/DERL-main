"""
解析 Pegasus DAX 格式的 XML 文件，生成 DAG 结构

DAX 文件格式说明：
    - <job> 标签定义任务，包含 id、name、version、runtime 等属性
    - <child> 和 <parent> 标签定义任务之间的依赖关系

数据模型：
    - 每个任务有 input / output 两个总大小（单位 bytes）
    - 入口任务（无父）：input 独立采样 [min, max]
    - 非入口任务：input = 所有父任务 output 的最大值
    - 所有任务：output 独立采样 [min, max]
"""

import random
import networkx as nx
from xml.etree import ElementTree as ET
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


class Task:
    """表示工作流中的单个任务节点（解析时使用）"""

    def __init__(self, task_id: int, xml_id: str):
        self.task_id = task_id
        self.xml_id = xml_id
        self.parents: List['Task'] = []
        self.children: List['Task'] = []
        self.runtime: float = 0.0


def generate_workflow_from_xml(path: str) -> Tuple[Optional[nx.DiGraph], Optional[Dict], Optional[Dict]]:
    """
    从 DAX 格式的 XML 文件解析生成工作流

    返回:
        graph: networkx.DiGraph，节点ID = 任务内部ID
        runtime_map: {任务ID: runtime}
        task_name_map: {任务ID: xml 中的 id 字符串}
    """
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        ns = {'dax': 'http://pegasus.isi.edu/schema/DAX'}

        xml_id_to_task: Dict[str, Task] = {}
        task_list: List[Task] = []
        job_id_counter = 0

        # 第一遍：创建任务节点，解析 runtime
        for node in root.findall('dax:job', ns):
            xml_id = node.get('id')
            runtime = float(node.get('runtime', 0))

            task = Task(job_id_counter, xml_id)
            task.runtime = runtime
            xml_id_to_task[xml_id] = task
            task_list.append(task)

            job_id_counter += 1

        # 第二遍：构建依赖关系
        for child_node in root.findall('dax:child', ns):
            child_xml_id = child_node.get('ref')
            if child_xml_id not in xml_id_to_task:
                continue

            child_task = xml_id_to_task[child_xml_id]

            for parent_node in child_node.findall('dax:parent', ns):
                parent_xml_id = parent_node.get('ref')
                if parent_xml_id in xml_id_to_task:
                    parent_task = xml_id_to_task[parent_xml_id]
                    parent_task.children.append(child_task)
                    child_task.parents.append(parent_task)

        # 构建 networkx 图和属性映射
        G = nx.DiGraph()
        runtime_map: Dict[int, float] = {}
        task_name_map: Dict[int, str] = {}

        for task in task_list:
            task_id = task.task_id
            G.add_node(task_id)
            runtime_map[task_id] = task.runtime
            task_name_map[task_id] = task.xml_id

            for parent in task.parents:
                G.add_edge(parent.task_id, task_id)

        return G, runtime_map, task_name_map

    except Exception as e:
        print(f"解析失败: {e}")
        return None, None, None


def build_edge_ai_pipeline(n_camera=3, n_pre=3, n_fusion=1, n_detect=1, n_track=1, n_decision=1):
    G = nx.DiGraph()
    cams   = [f"Cam{i}"    for i in range(n_camera)]
    pres   = [f"Pre{i}"    for i in range(n_pre)]
    fusion = [f"Fusion{i}" for i in range(n_fusion)]
    detect = [f"Detect{i}" for i in range(n_detect)]
    track  = [f"Track{i}"  for i in range(n_track)]
    dec    = [f"Dec{i}"    for i in range(n_decision)]
    G.add_nodes_from(cams + pres + fusion + detect + track + dec)
    for i, c in enumerate(cams):
        G.add_edge(c, pres[i % n_pre])
    for p in pres:
        for f in fusion:
            G.add_edge(p, f)
    for f in fusion:
        for d in detect:
            G.add_edge(f, d)
    for d in detect:
        for t in track:
            G.add_edge(d, t)
    for t in track:
        for x in dec:
            G.add_edge(t, x)
    return G


def build_microservice_dag():
    G = nx.DiGraph()
    G.add_edges_from([
        ("Gateway", "Auth"), ("Gateway", "PostSvc"), ("Gateway", "MediaSvc"),
        ("Auth", "UserSvc"), ("UserSvc", "Timeline"), ("PostSvc", "Timeline"),
        ("MediaSvc", "Store"), ("MediaSvc", "Timeline"), ("Store", "Timeline"),
        ("Timeline", "Frontend"),
    ])
    return G


def relabel_to_int(G):
    mapping = {name: i for i, name in enumerate(G.nodes())}
    return nx.relabel_nodes(G, mapping)


def get_all_entry_jobs(graph: nx.DiGraph) -> List[int]:
    """获取所有入口任务（无父任务）"""
    return [node for node in graph.nodes() if not list(graph.predecessors(node))]


def get_all_exit_jobs(graph: nx.DiGraph) -> List[int]:
    """获取所有出口任务（无子任务）"""
    return [node for node in graph.nodes() if not list(graph.successors(node))]


def sample_task_io_by_topology(graph, target_min_mb=5220, target_max_mb=20680, rng=None):
    if rng is None:
        rng = random   # 兼容旧调用

    target_min_bytes = int(target_min_mb * 1024 * 1024)
    target_max_bytes = int(target_max_mb * 1024 * 1024)

    input_map = {}
    output_map = {}

    for task_id in nx.topological_sort(graph):
        predecessors = list(graph.predecessors(task_id))

        if len(predecessors) == 0:
            total_in = rng.uniform(target_min_bytes, target_max_bytes)
        else:
            total_in = max(output_map.get(p, 0) for p in predecessors)

        input_map[task_id] = int(total_in)
        output_map[task_id] = int(rng.uniform(target_min_bytes, target_max_bytes))

    return input_map, output_map


# ============================================================
#  测试代码
# ============================================================
if __name__ == "__main__":
    import os

    G = relabel_to_int(build_edge_ai_pipeline())
    plt.figure(figsize=(8, 5))
    nx.draw(G, with_labels=True, node_color='lightblue',
            node_size=800, arrows=True, arrowsize=15)
    plt.title("Edge AI Pipeline DAG")
    plt.show()

    test_files = [
        "../dax/CyberShake_30.xml",
    ]

    base_path = os.path.join(os.path.dirname(__file__), "dax")
    if not os.path.exists(base_path):
        base_path = "."

    print("=" * 80)
    print("工作流 XML 解析与任务级 IO 采样测试（拓扑继承取最大值）")
    print("=" * 80)

    for filename in test_files:
        filepath = os.path.join(base_path, filename)
        if not os.path.exists(filepath):
            print(f"\n[跳过] 文件不存在: {filepath}")
            continue

        print(f"\n{'=' * 80}")
        print(f"测试文件: {filename}")
        print("=" * 80)

        G, runtime_map, task_name_map = generate_workflow_from_xml(filepath)

        if G is None:
            print("  解析失败！")
            continue

        print(f"  任务数量: {G.number_of_nodes()}")
        print(f"  依赖边数: {G.number_of_edges()}")
        print(f"  入口任务: {get_all_entry_jobs(G)}")
        print(f"  出口任务: {get_all_exit_jobs(G)}")

        # ============================================================
        #  任务级 IO 采样
        # ============================================================
        print("\n  " + "-" * 40)
        print("  任务级 IO 采样结果（拓扑继承取最大值）")
        print("  " + "-" * 40)

        input_map, output_map = sample_task_io_by_topology(G, target_min_mb=5220, target_max_mb=20680)

        # ============================================================
        #  详细统计
        # ============================================================
        print("\n  " + "-" * 40)
        print("  各任务 input 明细（前10个最大的）")
        print("  " + "-" * 40)

        input_details = [(t, v / (1024 * 1024)) for t, v in input_map.items()]
        input_details.sort(key=lambda x: x[1], reverse=True)

        for task_id, total in input_details[:10]:
            status = "✅" if 5220 <= total <= 20680 else "❌"
            print(f"    任务 {task_id:3d}: {total:10.2f} Mb {status}")

        print("\n  " + "-" * 40)
        print("  各任务 output 明细（前10个最大的）")
        print("  " + "-" * 40)

        output_details = [(t, v / (1024 * 1024)) for t, v in output_map.items()]
        output_details.sort(key=lambda x: x[1], reverse=True)

        for task_id, total in output_details[:10]:
            status = "✅" if 5220 <= total <= 20680 else "❌"
            print(f"    任务 {task_id:3d}: {total:10.2f} Mb {status}")

        # ============================================================
        #  拓扑继承验证（前10条边）
        # ============================================================
        print("\n  " + "-" * 40)
        print("  拓扑继承验证（前10条边）")
        print("  " + "-" * 40)

        count = 0
        for parent, child in G.edges():
            print(f"    {parent} → {child}: "
                  f"父 output {output_map[parent] / (1024 * 1024):.2f} Mb, "
                  f"子 input {input_map[child] / (1024 * 1024):.2f} Mb")
            count += 1
            if count >= 10:
                break

        print("\n  ✓ 解析完成")

    print("\n" + "=" * 80)
    print("测试结束")
    print("=" * 80)
