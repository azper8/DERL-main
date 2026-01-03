import numpy as np
import random
from scipy.stats import weibull_min


class FaultGenerator:
    """边缘节点故障时间窗生成器（秒级时间单位）

    功能：为虚拟机生成随机故障时间窗，时间精度为秒
    核心算法：
      1. 基于泊松过程生成故障事件时间点（秒级）
      2. 威布尔分布拟合故障持续时间（秒）
      3. 故障时间窗合并处理（解决重叠问题）

    时间单位：所有时间参数均以秒为单位
    """

    # 故障类型分布常数 (基于全球边缘计算故障统计报告)
    FAULT_TYPES = {
        # 格式: [故障类型: (发生概率, 持续时间范围(秒), 修复时间分布参数]
        # "transient": (0.68, (1, 30), (0.5, 2.0)),  # 瞬时故障（1-30秒）
        # "short_term": (0.25, (30, 300), (1.2, 50)),  # 短期故障（30-300秒）
        # "long_term": (0.07, (300, 1800), (1.5, 300))  # 长期故障（300-1800秒）
        "transient": (1, (1, 10), (0.8, 5)),  # 瞬时故障（1-10秒）
    }

    def __init__(self, device_type: str, simulation_seconds: int):
        """初始化故障生成器（秒级时间单位）

        参数：
          device_type: 设备类型
              'industrial' - 工业固定节点
              'mobile' - 移动设备节点
              'sensor' - 低功耗传感器节点
          simulation_seconds: 模拟运行总时长（秒）

        属性：
          fault_rate: 设备年故障率（基于类型）
          time_range: 模拟时间范围（秒）
          fault_events: 生成的故障事件列表
        """
        # 基于设备类型的年故障率（工业实践数据）
        FAULT_RATES = {
            'industrial': 0.25,  # 25%年故障率
            'mobile': 0.60,  # 60%年故障率（移动性增加风险）
            'sensor': 0.35  # 35%年故障率（环境敏感性）
        }

        if device_type not in FAULT_RATES:
            raise ValueError(f"无效设备类型。可选: {list(FAULT_RATES.keys())}")

        self.device_type = device_type
        self.fault_rate = FAULT_RATES[device_type]
        self.time_range = simulation_seconds  # 总模拟秒数
        self.fault_events = []  # 存储故障事件: [(开始时间, 结束时间), ...]

    def generate_fault_events(self):
        """生成故障事件序列（秒级精度）

        步骤：
          1. 计算预期故障次数（泊松分布）
          2. 生成故障发生时间点（均匀分布在模拟时间段内）
          3. 确定故障类型和持续时间（威布尔分布）
          4. 构建故障时间窗（秒级）

        注意：故障时间窗可能重叠，需后续合并处理
        """
        # 1. 计算预期故障次数 - 泊松分布λ参数 = (年故障率/年秒数)*模拟秒数
        SECONDS_PER_YEAR = 365 * 24 * 3600  # 一年总秒数
        # lambda_rate = (self.fault_rate / SECONDS_PER_YEAR) * self.time_range
        lambda_rate = (self.fault_rate / SECONDS_PER_YEAR) * 2e5 * self.time_range
        expected_events = np.random.poisson(lambda_rate)

        # 2. 生成故障发生时间点（均匀分布在模拟周期内）
        occurrence_times = sorted(
            random.uniform(0, self.time_range)
            for _ in range(expected_events)
        )

        # 3. 对每个故障事件生成时间窗（秒级）
        for t in occurrence_times:
            # 选择故障类型（基于预设概率分布）
            fault_type = random.choices(
                list(self.FAULT_TYPES.keys()),
                weights=[prob for prob, *_ in self.FAULT_TYPES.values()]
            )[0]

            # 获取故障类型参数
            prob, (min_dur, max_dur), (shape, scale) = self.FAULT_TYPES[fault_type]

            # 生成故障持续时间（秒） - 威布尔分布
            # 计算累积概率范围
            low_prob = weibull_min.cdf(min_dur, shape, scale=scale)
            high_prob = weibull_min.cdf(max_dur, shape, scale=scale)
            # 生成[low_prob, high_prob]范围内的均匀分布样本
            u = np.random.uniform(low_prob, high_prob)
            # 使用PPF转换回威布尔分布
            duration =  weibull_min.ppf(u, shape, scale=scale)

            # 故障结束时间 = 开始时间 + 持续时长（秒）
            end_time = t + duration

            # 存储故障事件（开始时间，结束时间，故障类型）
            self.fault_events.append({
                'start': t,  # 故障开始时间（秒）
                'end': end_time,  # 故障结束时间（秒）
                'duration': duration,  # 故障持续时间（秒）
                'type': fault_type  # 故障类型
            })

        # 4. 合并重叠时间窗（重要：避免重复计次）
        self._merge_overlapping_windows()
        return self.fault_events

    def _merge_overlapping_windows(self):
        """合并重叠故障时间窗（秒级时间处理）

        算法逻辑：
          1. 按故障开始时间排序
          2. 遍历事件列表，检查时间窗重叠
          3. 若重叠则合并为一个连续时间窗（秒级）

        示例：
          原事件: [(100s, 200s), (150s, 250s)] -> 合并为 [(100s, 250s)]
        """
        if not self.fault_events:
            return

        # 按开始时间排序（保证合并算法正确性）
        sorted_events = sorted(self.fault_events, key=lambda x: x['start'])
        merged_events = [sorted_events[0]]

        for current in sorted_events[1:]:
            last = merged_events[-1]

            # 检查时间窗重叠：当前事件开始时间 <= 上一个事件结束时间
            if current['start'] <= last['end']:
                # 扩展合并窗口的结束时间为较大值（秒级）
                last['end'] = max(last['end'], current['end'])
                # 更新持续时间和类型标记（取最严重类型）
                last['duration'] = last['end'] - last['start']
                # if current['type'] == 'long_term' or last['type'] != 'long_term':
                if current['type'] == 'short_term' or last['type'] != 'short_term':
                    last['type'] = current['type']
            else:
                # 无重叠，添加新事件
                merged_events.append(current)

        self.fault_events = merged_events

    def get_fault_windows(self):
        """获取故障时间窗列表（秒级时间单位）

        返回格式:
          [(start_time1, end_time1), (start_time2, end_time2), ...]
          所有时间值均为秒

        典型应用场景：
          调度器通过此接口获取节点不可用时段（秒级精度）
        """
        if not self.fault_events:
            self.generate_fault_events()
        return [(e['start'], e['end'], e['type']) for e in self.fault_events]

    def get_fault_summary(self):
        """生成故障统计摘要（秒级单位）

        返回：
          dict: 包含故障次数、总时长(秒)、各类型分布等统计信息
        """
        if not self.fault_events:
            return {}

        summary = {
            'total_events': len(self.fault_events),
            'total_duration': sum(e['duration'] for e in self.fault_events),
            'type_distribution': {},
            'downtime_ratio': None
        }

        # 按类型统计故障信息（秒级）
        for fault_type in self.FAULT_TYPES:
            type_events = [e for e in self.fault_events if e['type'] == fault_type]
            summary['type_distribution'][fault_type] = {
                'count': len(type_events),
                'total_duration': sum(e['duration'] for e in type_events),
                'avg_duration': (sum(e['duration'] for e in type_events) / len(type_events)
                                 if type_events else 0)
            }

        # 计算故障时间占比（秒级）
        summary['downtime_ratio'] = summary['total_duration'] / self.time_range
        return summary


if __name__ == '__main__':
    # 1. 设置模拟参数
    SIMULATION_SECONDS = 86400  # 模拟1天（86400秒）
    DEVICE_TYPE = 'mobile'  # 设备类型：'industrial', 'mobile' 或 'sensor'

    # 2. 创建故障生成器实例
    fault_gen = FaultGenerator(
        device_type=DEVICE_TYPE,
        simulation_seconds=SIMULATION_SECONDS
    )

    count = 0
    for i in range(1000):
        # 3. 生成故障事件
        fault_events = fault_gen.generate_fault_events()

        # 4. 获取故障时间窗列表
        fault_windows = fault_gen.get_fault_windows()

        # 5. 获取故障统计摘要
        log = fault_gen.get_fault_summary()

        # 6. 输出结果
        print(f"生成的故障时间窗数量: {len(fault_windows)}")
        if log:
            print(fault_windows)
            print(f"总故障时间: {log['total_duration']:.2f}秒")
            print(f"设备可用率: {(1 - log['downtime_ratio']) * 100:.2f}%")
            count += 1
            fault_gen.fault_events = []
    print(count)
