import networkx as nx
import matplotlib.pyplot as plt
from xml.etree import ElementTree as ET


class Task:
    def __init__(self, task_id):
        """表示工作流中的单个任务节点
        Args:
            task_id (int): 任务的唯一标识符
        """
        self.task_id = task_id   # 任务ID
        self.parents = []        # 存储父任务对象列表（前驱节点）
        self.children = []       # 存储子任务对象列表（后继节点）


def generate_workflow_job_from_xml_file(path):
    """从XML文件解析生成工作流DAG（有向无环图）
    Args:
        path (str): DAX格式的XML文件路径
    Returns:
        nx.DiGraph: 用networkx表示的有向无环图
    """
    try:
        tree = ET.parse(path)   # 解析XML文件
        root = tree.getroot()   # 获取根节点

        # 初始化数据结构
        mName2Task = {}         # 映射：XML节点ID -> Task对象
        job_id_starts_from = 0  # 任务ID计数器
        task_list = []          # 存储所有Task对象

        # 定义XML命名空间（Pegasus DAX格式需要）
        namespace = {'dax': 'http://pegasus.isi.edu/schema/DAX'}

        # 第一遍遍历：创建所有任务节点
        for node in root.findall('dax:job', namespace):
            node_id = node.get('id')      # 获取XML中的任务ID
            node_name = node.get('name')  # 获取任务名称（未使用）

            # 创建Task对象并建立映射
            task = Task(job_id_starts_from)
            job_id_starts_from += 1
            mName2Task[node_id] = task
            task_list.append(task)

        # 第二遍遍历：构建任务依赖关系
        for node in root.findall('dax:child', namespace):
            child_name = node.get('ref')  # 获取子任务ID
            if child_name in mName2Task:
                child_task = mName2Task[child_name]

                # 遍历当前子任务的所有父任务
                for parent in node.findall('dax:parent', namespace):
                    parent_name = parent.get('ref')
                    if parent_name in mName2Task:
                        parent_task = mName2Task[parent_name]
                        # 建立双向连接
                        parent_task.children.append(child_task)
                        child_task.parents.append(parent_task)

        # 清理临时数据节省内存
        mName2Task.clear()

        # 将Task列表转换为networkx有向图
        G = nx.DiGraph()
        for task in task_list:
            G.add_node(task.task_id)
            for parent in task.parents:
                G.add_edge(parent.task_id, task.task_id)
        return G

    except ET.ParseError:
        print("XML Parsing Error; Please make sure your XML file is valid.")
        return []

def visualize_workflow(tasks):
    """可视化工作流DAG
    Args:
        tasks (list[Task]): Task对象列表
    """
    G = nx.DiGraph()

    # 构建networkx图结构
    for task in tasks:
        G.add_node(task.task_id)
        for parent in task.parents:
            G.add_edge(parent.task_id, task.task_id)

    # 使用环形布局绘制图形
    pos = nx.circular_layout(G)

    # 验证是否为有向无环图
    is_dag = nx.is_directed_acyclic_graph(G)
    if is_dag:
        print("G 是有向无环图（DAG）")
    else:
        print("G 不是有向无环图（DAG）")

    # 绘制图形参数设置
    nx.draw(G,
            pos,
            with_labels=True,      # 显示节点标签
            node_size=700,         # 节点大小
            node_color="skyblue",  # 节点颜色
            font_size=8,           # 标签字体大小
            font_color="black",    # 标签颜色
            font_weight="bold",    # 字体加粗
            arrowsize=10)          # 箭头大小

    plt.title("Workflow Job Visualization")
    plt.show()
