class Server:
    def __init__(self, es_id, bandwidth, pa, server_type, name):
        """
        服务器/设备基类（可表示移动设备、边缘服务器或中心服务器）

        参数:
            es_id (int): 服务器唯一标识符
            bandwidth (float): 带宽（单位：bps，比特每秒）
            pa (float): 处理能力（单位：Hz，处理器主频）
            server_type (str): 服务器类型（如 "MobileDevice"/"EdgeServer"/"CenterServer"）
            name (str): 服务器名称（如 "EdgeServer_1"）
        """
        self.id = es_id                   # 服务器唯一ID
        self.bandwidth = bandwidth        # 网络带宽（单位：bps）
        self.pa = pa                      # 处理能力（Processing Ability，单位：Hz）
        self.available = True
        self.record = []                  # 已完成任务的历史记录（存储已完成的Job对象）
        self.isBusy = False               # 当前是否正在处理任务（布尔值）
        self.trans_list = []
        self.wait_list = []               # 等待队列（存储等待处理的Job对象）
        self.processing_job = None        # 当前正在处理的任务（Job对象或None）
        self.server_type = server_type    # 服务器类型（用于区分设备类型）
        self.name = name                  # 服务器名称（用于可视化或日志）
        self.fault_windows = None