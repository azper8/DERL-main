from copy import deepcopy


def agent_policy(agent_ids, policy):
    """
    为多个智能体创建策略字典，每个智能体获得策略的深拷贝
    :param agent_ids: 智能体ID列表
    :param policy: 基础策略对象
    :return: 字典，键为智能体ID，值为策略的深拷贝
    """
    group = {}  # 初始化策略字典
    for agent_id in agent_ids:  # 遍历所有智能体ID
        # 为每个智能体创建策略的深拷贝(完全独立的对象)
        group[agent_id] = deepcopy(policy)
    return group