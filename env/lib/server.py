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
        self.id = es_id  # 服务器唯一ID
        self.bandwidth = bandwidth  # 网络带宽（单位：bps）
        self.pa = pa  # 处理能力（Processing Ability，单位：Hz）
        self.available = True  # 是否可用（故障恢复用）
        self.record = []  # 已完成任务的历史记录
        self.isBusy = False  # 当前是否正在处理任务
        self.trans_list = []  # 传输队列（已废弃，保留兼容）
        self.wait_list = []  # 等待队列
        self.processing_job = None  # 当前正在处理的任务
        self.server_type = server_type  # 服务器类型
        self.name = name  # 服务器名称
        self.fault_windows = None  # 故障时间窗（故障恢复用）

        # ====== 新增：区域/距离层标识 ======
        # 用于服务器间带宽计算
        # - 边缘服务器: region = 0（表示本地）
        # - 云服务器: region = 128/256/512（表示不同距离层）
        self.region = 0

        # ====== 新增：网络波动相关 ======
        # bandwidth_factor: 当前时刻该服务器链路的带宽波动系数
        # 1.0 表示无波动；<1 表示带宽下降；>1 表示带宽提升
        # base_bandwidth: 保留名义带宽（便于统计/恢复）
        self.bandwidth_factor = 1.0
        self.base_bandwidth = bandwidth