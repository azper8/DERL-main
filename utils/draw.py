# import math
# import matplotlib.cm as cm
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
#
#
# def visualize_job_timeline(env, name):
#     """
#     绘制服务器和本地设备的任务执行甘特图：
#     - 横轴：时间轴（自动取整十）
#     - 纵轴：设备类型+编号（如 "Server 0", "Device 1"）
#     - 每个任务用三段式填充（上传/处理/下载）
#     - 支持动态调整图形大小
#     """
#     # 合并所有设备（服务器+本地设备）
#     all_devices = []
#     e_num = 0
#     c_num = 0
#     if hasattr(env, 'servers'):
#         for s in env.servers:
#             if s.bandwidth == 1024**3:
#                 all_devices.append((f"Edge {e_num}", s))
#                 e_num += 1
#             else:
#                 all_devices.append((f"Cloud {c_num}", s))
#                 c_num += 1
#     if hasattr(env, 'devices'):
#         all_devices.extend([(f"Device {d.id}", d) for d in env.devices])
#
#     # 自定义排序规则
#     def custom_sort_key(device_item):
#         """
#         排序规则:
#         1. 先按类型排序: Cloud > Edge > Device
#         2. 相同类型内按ID数字从大到小排序
#         """
#         name, _ = device_item
#         parts = name.split()
#         device_type = parts[0]
#         device_id = int(parts[1])
#
#         # 类型优先级：Cloud > Edge > Device
#         type_priority = {
#             'Cloud': 0,
#             'Edge': 1,
#             'Device': 2
#         }
#
#         # 返回元组，先按类型优先级排序，再按ID倒序
#         return (type_priority.get(device_type, 99), -device_id)
#
#     # 应用排序规则
#     all_devices = sorted(all_devices, key=custom_sort_key, reverse=True)
#
#     # 打印调试信息 - 检查设备列表
#     # print(f"Found {len(all_devices)} devices:")
#     # for name, device in all_devices:
#     #     record_count = len(device.record) if hasattr(device, 'record') else 0
#     # print(f"  {name}: {record_count} jobs in record")
#
#     # 动态生成颜色（支持任意数量的workflow）
#     num_workflows = len(env.workflows) if hasattr(env, 'workflows') else 1
#     colormap = cm.get_cmap('tab20', num_workflows)
#     workflow_colors = [colormap(i) for i in range(num_workflows)]
#
#     # 计算最大时间（取整十）
#     max_time = 0
#     valid_jobs_count = 0
#     for _, device in all_devices:
#         if hasattr(device, 'record'):
#             for job in device.record:
#                 if hasattr(job, 'finishTime') and job.finishTime is not None:
#                     max_time = max(max_time, job.finishTime)
#                     valid_jobs_count += 1
#     max_time_rounded = math.ceil(max(max_time, 100) / 10) * 10
#
#     # 打印调试信息 - 检查任务数据
#     # print(f"Found {valid_jobs_count} valid jobs with processEndTime")
#     # print(f"Max time in records: {max_time}, rounded to: {max_time_rounded}")
#
#     # 动态调整图形大小
#     num_devices = len(all_devices)
#     fig_width = max(15, min(30, max_time_rounded / 10))
#     fig_height = max(6, num_devices * 0.8)
#     fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='white')
#
#     for spine in ax.spines.values():
#         spine.set_linewidth(2)  # 设置坐标轴边框宽度
#
#     # 动态调整block高度和间距
#     block_height = min(0.8, max(0.4, 0.8 - num_devices * 0.01))
#     spacing = block_height * 0.5
#
#     # 计算纵坐标位置（倒序排列，Server 0在最上方）
#     y_positions = {}
#     for i, (device_name, _) in enumerate(all_devices):
#         y_positions[device_name] = i * (block_height + spacing) + block_height / 2
#
#     # 绘制每个设备的任务
#     jobs_drawn = 0
#     for device_name, device in all_devices:
#         print(device_name)
#         y_pos = y_positions[device_name]
#         if not hasattr(device, 'record'):
#             continue
#
#         for job in device.record:
#             # 检查必要的时间属性
#             required_attrs = ['processStartTime', 'processEndTime', 'id']
#             if not all(hasattr(job, attr) for attr in required_attrs):
#                 print(f"Skipping job in {device_name} - missing required attributes")
#                 continue
#             if job.processStartTime is None or job.processEndTime is None:
#                 print(f"Skipping job {job.id} in {device_name} - null timestamps")
#                 continue
#
#             # 计算任务总持续时间
#             total_duration = job.processEndTime - job.processStartTime
#             # 1. 绘制任务边框
#             job_box = patches.Rectangle(
#                 (job.processStartTime, y_pos - block_height / 2),
#                 total_duration,
#                 block_height,
#                 fill=False, edgecolor='black', linewidth=1.2,
#                 zorder=3
#             )
#             ax.add_patch(job_box)
#
#             # 2. 处理阶段填充
#             color_idx = int(job.workflow_id) % len(workflow_colors) if hasattr(job, 'workflow_id') else 0
#             process_color = (*workflow_colors[color_idx][:3], 0.5)
#
#             process = patches.Rectangle(
#                 (job.processStartTime, y_pos - block_height / 2),
#                 total_duration,
#                 block_height,
#                 facecolor=process_color, edgecolor='none', linewidth=0, zorder=1
#             )
#             ax.add_patch(process)
#             jobs_drawn += 1
#
#             # 3. 上传阶段（仅服务器任务）
#             if ("Edge" in device_name or "Cloud" in device_name) and hasattr(job, 'readyTime') and hasattr(job, 'arriveTime'):
#                 if job.readyTime is not None and job.arriveTime is not None:
#                     upload_duration = job.arriveTime - job.readyTime
#                     if upload_duration > 0:
#                         upload = patches.Rectangle(
#                             (job.readyTime, y_pos - block_height / 2),
#                             upload_duration,
#                             block_height,
#                             facecolor='none', edgecolor='red', linewidth=0.5,
#                             hatch='///', zorder=2
#                         )
#                         ax.add_patch(upload)
#
#             # 4. 下载阶段（仅服务器任务）
#             if ("Edge" in device_name or "Cloud" in device_name) and hasattr(job, 'processEndTime') and hasattr(job, 'finishTime'):
#                 if job.processEndTime is not None and job.finishTime is not None:
#                     download_duration = job.finishTime - job.processEndTime
#                     if download_duration > 0:
#                         download = patches.Rectangle(
#                             (job.processEndTime, y_pos - block_height / 2),
#                             download_duration,
#                             block_height,
#                             facecolor='none', edgecolor='green', linewidth=0.5,
#                             hatch='\\\\\\', zorder=2
#                         )
#                         ax.add_patch(download)
#
#             # 5. 添加任务标签
#             label_x = job.processStartTime + total_duration / 2
#             ax.text(
#                 label_x, y_pos, f"t{job.id}",
#                 ha='center', va='center', fontsize=18,
#                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0),
#                 zorder=4
#             )
#
#     # 打印调试信息 - 总结
#     # print(f"Successfully drew {jobs_drawn} jobs on the timeline")
#
#     # --- 坐标轴设置 ---
#     ax.set_xlim(0, max_time_rounded)
#     ax.set_ylim(-block_height, max(y_positions.values()) + block_height)
#
#     # 设置y轴标签
#     y_ticks = [y_positions[name] for name, _ in all_devices]
#     ax.set_yticks(y_ticks)
#     ax.set_yticklabels([name for name, _ in all_devices])
#
#     # ax.set_xlabel('Time', fontsize=20, fontweight='bold')
#     # ax.set_ylabel('Processor', fontsize=20, fontweight='bold')
#     ax.tick_params(axis='x', labelsize=20)  # x轴刻度标签大小
#     ax.tick_params(axis='y', labelsize=20)  # y轴刻度标签大小
#     # ax.set_title('Task Execution Timeline', fontsize=20, fontweight='bold')
#
#     # --- 图例 ---
#     legend_elements = [
#         patches.Patch(facecolor='white', edgecolor='red', hatch='//', label='Upload Phase'),
#         patches.Patch(facecolor='white', edgecolor='green', hatch='\\\\', label='Download Phase')
#     ]
#     for i, color in enumerate(workflow_colors):
#         # legend_elements.append(patches.Patch(facecolor=color, label=f'Workflow {i}'))
#         legend_elements.append(patches.Patch(facecolor=color, label=f'Execution Phase'))
#     # ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
#     ax.legend(handles=legend_elements, loc='upper right', fontsize=14)
#
#     # --- 网格和保存 ---
#     ax.grid(True, axis='x', linestyle='--', alpha=0.3)
#     plt.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.1)
#
#     # 确保保存目录存在
#     import os
#     save_path = f'analyse/plots/{name}.png'
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#
#     plt.savefig(save_path, format='png', bbox_inches='tight', dpi=300)
#     print(f"\nTimeline saved to {save_path}")
#     plt.close()
#


import math
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import colorsys


def visualize_job_timeline(env, name):
    """
    绘制服务器和本地设备的任务执行甘特图：
    - 横轴：时间轴（自动取整十）
    - 纵轴：设备类型+编号（如 "Server 0", "Device 1"）
    - 每个任务用三段式填充（上传/处理/下载）
    - 支持动态调整图形大小
    """
    # 合并所有设备（服务器+本地设备）
    all_devices = []
    e_num = 0
    c_num = 0
    if hasattr(env, 'servers'):
        for s in env.servers:
            if s.bandwidth == 1024 ** 3:
                all_devices.append((f"E{e_num}", s))
                e_num += 1
            else:
                all_devices.append((f"C{c_num}", s))
                c_num += 1
    if hasattr(env, 'devices'):
        all_devices.extend([(f"D{d.id}", d) for d in env.devices])

    # 自定义排序规则
    def custom_sort_key(device_item):
        """
        排序规则:
        1. 先按类型排序: Cloud > Edge > Device
        2. 相同类型内按ID数字从大到小排序
        """
        name, _ = device_item
        parts = name.split()
        device_type = str(name)[0]
        device_id = int(str(name)[1])

        # 类型优先级：Cloud > Edge > Device
        type_priority = {
            'C': 0,
            'E': 1,
            'D': 2
        }

        # 返回元组，先按类型优先级排序，再按ID倒序
        return (type_priority.get(device_type, 99), -device_id)

    # 应用排序规则
    all_devices = sorted(all_devices, key=custom_sort_key, reverse=True)

    # 打印调试信息 - 检查设备列表
    # print(f"Found {len(all_devices)} devices:")
    # for name, device in all_devices:
    #     record_count = len(device.record) if hasattr(device, 'record') else 0
    # print(f"  {name}: {record_count} jobs in record")

    # 动态生成颜色（支持任意数量的workflow）
    num_workflows = len(env.workflows) if hasattr(env, 'workflows') else 1
    colormap = cm.get_cmap('tab20', num_workflows)
    workflow_colors = [colormap(i) for i in range(num_workflows)]

    # 计算最大时间（取整十）
    max_time = 0
    valid_jobs_count = 0
    for _, device in all_devices:
        if hasattr(device, 'record'):
            for job in device.record:
                if hasattr(job, 'finishTime') and job.finishTime is not None:
                    max_time = max(max_time, job.finishTime)
                    valid_jobs_count += 1
    max_time_rounded = math.ceil(max(max_time, 100) / 10) * 10

    # 打印调试信息 - 检查任务数据
    # print(f"Found {valid_jobs_count} valid jobs with processEndTime")
    # print(f"Max time in records: {max_time}, rounded to: {max_time_rounded}")

    # 动态调整图形大小
    num_devices = len(all_devices)
    fig_width = max(15, min(30, max_time_rounded / 10))
    fig_height = max(6, num_devices * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='white')

    # 设置坐标图背景为浅灰色 (233, 233, 233)
    ax.set_facecolor('#E9E9E9')  # RGB 233, 233, 233 的十六进制

    # 移除坐标轴的外边框（上、右、左、下边框）
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # 或者更简洁地全部移除
    # for spine in ax.spines.values():
    #     spine.set_visible(False)

    # 动态调整block高度和间距
    block_height = min(0.8, max(0.4, 0.8 - num_devices * 0.01))
    spacing = block_height * 0.5

    # 计算纵坐标位置（倒序排列，Server 0在最上方）
    y_positions = {}
    for i, (device_name, _) in enumerate(all_devices):
        y_positions[device_name] = i * (block_height + spacing) + block_height / 2

    # 绘制每个设备的任务
    jobs_drawn = 0
    for device_name, device in all_devices:
        # print(device_name)
        y_pos = y_positions[device_name]
        if not hasattr(device, 'record'):
            continue

        for job in device.record:
            # 检查必要的时间属性
            required_attrs = ['processStartTime', 'processEndTime', 'id']
            if not all(hasattr(job, attr) for attr in required_attrs):
                print(f"Skipping job in {device_name} - missing required attributes")
                continue
            if job.processStartTime is None or job.processEndTime is None:
                print(f"Skipping job {job.id} in {device_name} - null timestamps")
                continue

            # 计算任务总持续时间
            total_duration = job.processEndTime - job.processStartTime

            # 获取工作流颜色索引
            color_idx = int(job.workflow_id) % len(workflow_colors) if hasattr(job, 'workflow_id') else 0

            # 获取基础颜色
            base_color = workflow_colors[color_idx]
            base_rgb = base_color[:3]  # 取RGB值，忽略alpha

            # 将RGB转换为HSV，调整亮度来获得更深颜色
            # 方法1：直接按比例变暗
            darker_factor = 0.7  # 70%的亮度
            border_color = tuple(c * darker_factor for c in base_rgb)

            # 方法2：使用HSV调整（更专业）
            # h, s, v = colorsys.rgb_to_hsv(*base_rgb)
            # border_color = colorsys.hsv_to_rgb(h, s, v * 0.7)  # 降低亮度

            # 处理阶段填充（使用原色，半透明）
            process_color = (*base_rgb, 0.5)

            # 1. 处理阶段填充（zorder=2）
            process = patches.Rectangle(
                (job.processStartTime, y_pos - block_height / 2),
                total_duration,
                block_height,
                facecolor=process_color, edgecolor='none', linewidth=0, zorder=2
            )
            ax.add_patch(process)
            jobs_drawn += 1

            # 2. 绘制任务边框（使用同色系更深的颜色，zorder=3）
            job_box = patches.Rectangle(
                (job.processStartTime, y_pos - block_height / 2),
                total_duration,
                block_height,
                fill=False, edgecolor=border_color, linewidth=1.5,
                zorder=3
            )
            ax.add_patch(job_box)

            # 3. 上传阶段（仅服务器任务）（zorder=4）
            if ("E" in device_name or "C" in device_name) and hasattr(job, 'readyTime') and hasattr(job, 'arriveTime'):
                if job.readyTime is not None and job.arriveTime is not None:
                    upload_duration = job.arriveTime - job.readyTime
                    if upload_duration > 0:
                        upload = patches.Rectangle(
                            (job.readyTime, y_pos - block_height / 2),
                            upload_duration,
                            block_height,
                            facecolor='none', edgecolor='red', linewidth=1.5, alpha=0.7,
                            hatch='///', zorder=4
                        )
                        ax.add_patch(upload)

            # 4. 下载阶段（仅服务器任务）（zorder=4）
            if ("E" in device_name or "C" in device_name) and hasattr(job, 'processEndTime') and hasattr(job, 'finishTime'):
                if job.processEndTime is not None and job.finishTime is not None:
                    download_duration = job.finishTime - job.processEndTime
                    if download_duration > 0:
                        download = patches.Rectangle(
                            (job.processEndTime, y_pos - block_height / 2),
                            download_duration,
                            block_height,
                            facecolor='none', edgecolor='green', linewidth=1.5, alpha=0.7,
                            hatch='\\\\\\', zorder=4
                        )
                        ax.add_patch(download)

            # 5. 添加任务标签（zorder=5，在最上层）
            label_x = job.processStartTime + total_duration / 2
            ax.text(
                label_x, y_pos, f"t{job.id}",
                ha='center', va='center', fontsize=18,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0),
                zorder=5
            )

    # 打印调试信息 - 总结
    # print(f"Successfully drew {jobs_drawn} jobs on the timeline")

    # --- 坐标轴设置 ---
    ax.set_xlim(0, max_time_rounded)
    ax.set_ylim(-block_height, max(y_positions.values()) + block_height)

    # 设置y轴标签
    y_ticks = [y_positions[name] for name, _ in all_devices]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([name for name, _ in all_devices])

    # ax.set_xlabel('Time', fontsize=20, fontweight='bold')
    # ax.set_ylabel('Processor', fontsize=20, fontweight='bold')
    ax.tick_params(axis='x', labelsize=20)  # x轴刻度标签大小
    ax.tick_params(axis='y', labelsize=20)  # y轴刻度标签大小
    # ax.set_title('Task Execution Timeline', fontsize=20, fontweight='bold')

    # --- 网格设置 ---
    # 添加白色网格线（放在最底层，zorder=0）
    ax.grid(True, axis='both', linestyle='-', linewidth=1.5, color='white', alpha=0.8, zorder=0)
    # 保留原有网格设置作为次要网格
    ax.grid(True, axis='x', linestyle='--', alpha=0.3, which='minor', zorder=0)

    # --- 图例 ---
    legend_elements = [
        patches.Patch(facecolor='white', edgecolor='red', hatch='///', label='Upload Phase'),
        patches.Patch(facecolor='white', edgecolor='green', hatch='\\\\\\', label='Download Phase')
    ]
    for i, color in enumerate(workflow_colors):
        # 为图例也使用深色边框
        base_rgb = color[:3]
        darker_color = tuple(c * 0.7 for c in base_rgb)
        legend_elements.append(
            patches.Patch(
                facecolor=(*base_rgb, 0.5),  # 半透明填充
                edgecolor=darker_color,  # 深色边框
                linewidth=1.5,
                label=f'Execution Phase'
            )
        )
    # ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=14)

    plt.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.1)

    # 确保保存目录存在
    import os
    save_path = f'analyse/plots/{name}.svg'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)
    print(f"\nTimeline saved to {save_path}")
    plt.close()