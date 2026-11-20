import math
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


def visualize_job_timeline(env, name):
    """
    绘制服务器和本地设备的任务执行甘特图：
    - 横轴：时间轴（自动取整十）
    - 纵轴：设备类型+编号（如 "Server 0", "Device 1"）
    - 每个任务用三段式填充（上传/处理/下载）
    - 上传 = arriveTime - readyTime
    - 处理 = processEndTime - processStartTime
    - 下载 = finishTime - processEndTime，只对出口任务绘制
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
        name, _ = device_item
        device_type = str(name)[0]
        device_id = int(str(name)[1])

        type_priority = {'C': 0, 'E': 1, 'D': 2}
        return (type_priority.get(device_type, 99), -device_id)

    all_devices = sorted(all_devices, key=custom_sort_key, reverse=True)

    # 动态生成颜色
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

    # 动态调整图形大小
    num_devices = len(all_devices)
    fig_width = max(15, min(30, max_time_rounded / 10))
    fig_height = max(6, num_devices * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='white')

    ax.set_facecolor('#E9E9E9')

    for spine in ax.spines.values():
        spine.set_visible(False)

    block_height = min(0.8, max(0.4, 0.8 - num_devices * 0.01))
    spacing = block_height * 0.5

    # 计算纵坐标位置
    y_positions = {}
    for i, (device_name, _) in enumerate(all_devices):
        y_positions[device_name] = i * (block_height + spacing) + block_height / 2

    # 绘制每个设备的任务
    jobs_drawn = 0
    for device_name, device in all_devices:
        y_pos = y_positions[device_name]
        if not hasattr(device, 'record'):
            continue

        for job in device.record:
            # 检查必要的时间属性
            if not all(hasattr(job, attr) for attr in ['processStartTime', 'processEndTime', 'id']):
                continue
            if job.processStartTime is None or job.processEndTime is None:
                continue

            # ---- 处理阶段 ----
            total_duration = job.processEndTime - job.processStartTime

            # 获取工作流颜色
            color_idx = int(job.workflow_id) % len(workflow_colors) if hasattr(job, 'workflow_id') else 0
            base_color = workflow_colors[color_idx]
            base_rgb = base_color[:3]

            darker_factor = 0.7
            border_color = tuple(c * darker_factor for c in base_rgb)
            process_color = (*base_rgb, 0.5)

            # 1. 处理阶段填充
            process = patches.Rectangle(
                (job.processStartTime, y_pos - block_height / 2),
                total_duration,
                block_height,
                facecolor=process_color, edgecolor='none', linewidth=0, zorder=2
            )
            ax.add_patch(process)
            jobs_drawn += 1

            # 2. 任务边框
            job_box = patches.Rectangle(
                (job.processStartTime, y_pos - block_height / 2),
                total_duration,
                block_height,
                fill=False, edgecolor=border_color, linewidth=1.5,
                zorder=3
            )
            ax.add_patch(job_box)

            # ---- 3. 上传/传输阶段（服务器任务）----
            # 上传 = arriveTime - readyTime
            # if ("E" in device_name or "C" in device_name):
            if True:
                if hasattr(job, 'readyTime') and hasattr(job, 'arriveTime'):
                    if job.readyTime is not None and job.arriveTime is not None:
                        upload_duration = job.arriveTime - job.readyTime
                        if upload_duration > 0:
                            upload = patches.Rectangle(
                                (job.readyTime, y_pos - block_height / 2),
                                upload_duration,
                                block_height,
                                facecolor='none', edgecolor='red', linewidth=1.5,
                                alpha=0.7, hatch='///', zorder=4
                            )
                            ax.add_patch(upload)

            # ---- 4. 下载阶段（只对出口任务）----
            # 下载 = finishTime - processEndTime
            if ("E" in device_name or "C" in device_name):
                if hasattr(job, 'processEndTime') and hasattr(job, 'finishTime'):
                    if job.processEndTime is not None and job.finishTime is not None:
                        # 判断是否是出口任务
                        workflow = env.workflows[job.workflow_id]
                        is_exit = (len(list(workflow.graph.successors(job.id))) == 0)

                        if is_exit:
                            download_duration = job.finishTime - job.processEndTime
                            if download_duration > 0:
                                download = patches.Rectangle(
                                    (job.processEndTime, y_pos - block_height / 2),
                                    download_duration,
                                    block_height,
                                    facecolor='none', edgecolor='green', linewidth=1.5,
                                    alpha=0.7, hatch='\\\\\\', zorder=4
                                )
                                ax.add_patch(download)

            # ---- 5. 任务标签 ----
            label_x = job.processStartTime + total_duration / 2
            ax.text(
                label_x, y_pos, f"t{job.id}",
                ha='center', va='center', fontsize=18,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0),
                zorder=5
            )

    # --- 坐标轴设置 ---
    ax.set_xlim(0, max_time_rounded)
    ax.set_ylim(-block_height, max(y_positions.values()) + block_height)

    y_ticks = [y_positions[name] for name, _ in all_devices]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([name for name, _ in all_devices])

    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    # 手动控制 X 轴刻度间隔
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(50))  # 主刻度间隔，按需改

    # --- 网格设置 ---
    ax.grid(True, axis='both', linestyle='-', linewidth=1.5, color='white', alpha=0.8, zorder=0)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3, which='minor', zorder=0)

    # --- 图例 ---
    legend_elements = [
        patches.Patch(facecolor='white', edgecolor='red', hatch='///', label='Upload Phase'),
        patches.Patch(facecolor='white', edgecolor='green', hatch='\\\\\\', label='Download Phase'),
    ]
    for i, color in enumerate(workflow_colors):
        base_rgb = color[:3]
        darker_color = tuple(c * 0.7 for c in base_rgb)
        legend_elements.append(
            patches.Patch(
                facecolor=(*base_rgb, 0.5),
                edgecolor=darker_color,
                linewidth=1.5,
                label=f'Execution Phase'
            )
        )
    ax.legend(handles=legend_elements, loc='lower right', fontsize=14)

    plt.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.1)

    save_path = f'analyse/plots/{name}.svg'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)
    print(f"\nTimeline saved to {save_path}")
    plt.close()