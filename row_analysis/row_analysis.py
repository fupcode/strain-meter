import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import fpc.pyplot_template
import os


def last_argmin(arr):
    min_value = np.min(arr)
    min_indices = np.where(arr == min_value)[0]
    last_min_index = max(min_indices)
    return last_min_index

def last_argmax(arr):
    max_value = np.max(arr)
    max_indices = np.where(arr == max_value)[0]
    last_max_index = max(max_indices)
    return last_max_index


def find_intersection(data, target_value, x_range):
    # Convert all inputs to float to avoid overflow when subtracting
    data = [int(x) for x in data]
    target_value = int(target_value)
    x_min, x_max = map(int, x_range)

    min_dist = float('inf')
    x_value = None
    prev_x, prev_y = None, None

    for cur_x in range(x_min, x_max + 1):
        if cur_x < len(data):
            cur_y = data[cur_x]
            if prev_y is not None and ((prev_y - target_value) * (cur_y - target_value) <= 0):
                if abs(cur_y - prev_y) < 1e-10:
                    approx_x = cur_x if abs(cur_y - target_value) < abs(prev_y - target_value) else prev_x
                else:
                    approx_x = (target_value - prev_y) * (cur_x - prev_x) / (cur_y - prev_y) + prev_x
                if abs(approx_x - (x_max + x_min) / 2) < min_dist:
                    min_dist = abs(approx_x - (x_max + x_min) / 2)
                    x_value = approx_x
            prev_x, prev_y = cur_x, cur_y

    return x_value


# 读取配置文件
with open("config.json", "r") as config_file:
    config = json.load(config_file)

# 读取图像
image_path = config["image_path"]
image = Image.open(image_path)
image_array = np.array(image)
distance = None

# 获取图像的红色通道
red_channel = image_array[0, :, 0]  # 仅取第一行红色通道
plt.figure(figsize=(10, 6))
plt.plot(red_channel, 'r-')

# 获取配置信息
left_edge = config["left_edge"]
right_edge = config["right_edge"]
default_width = config["default_width"]
left_width = config["left_width"] if config["left_width"] != 0 else default_width
right_width = config["right_width"] if config["right_width"] != 0 else default_width

# 分析左边缘
left_edge_values = red_channel[left_edge - left_width: left_edge + left_width]
left_max_value = np.max(left_edge_values)
left_min_value = np.min(left_edge_values)
left_max_x = left_edge - left_width + np.argmax(left_edge_values)
left_min_x = left_edge - left_width + last_argmin(left_edge_values)

left_edge_y = (left_max_value + left_min_value) / 2  # y坐标为最高点和最低点的中间值
left_edge_x = find_intersection(red_channel, left_edge_y, (left_min_x, left_max_x))  # 计算x坐标

# 分析右边缘
right_edge_values = red_channel[right_edge - right_width: right_edge + right_width]
right_max_value = np.max(right_edge_values)
right_min_value = np.min(right_edge_values)
right_max_x = right_edge - right_width + last_argmax(right_edge_values)
right_min_x = right_edge - right_width + np.argmin(right_edge_values)

right_edge_y = (right_max_value + right_min_value) / 2  # y坐标为最高点和最低点的中间值
right_edge_x = find_intersection(red_channel, right_edge_y, (right_max_x, right_min_x))  # 计算x坐标

# 绘制图像
if left_edge_x is None or right_edge_x is None:
    plt.axvline(x=left_edge, color='b', linestyle='--')
    plt.axvline(x=left_edge-left_width, color='b', linestyle='-')
    plt.axvline(x=left_edge+left_width, color='b', linestyle='-')

    plt.axvline(x=right_edge, color='g', linestyle='--')
    plt.axvline(x=right_edge-right_width, color='g', linestyle='-')
    plt.axvline(x=right_edge+right_width, color='g', linestyle='-')

    print(f"left_edge_x={left_edge_x}, right_edge_x={right_edge_x}")

else:
    plt.plot([left_max_x-left_width, left_max_x+left_width], [left_max_value, left_max_value], 'b-')
    plt.plot([left_min_x-left_width, left_min_x+left_width], [left_min_value, left_min_value], 'b-')
    # 标记最高点
    plt.plot(left_max_x, left_max_value, 'bo')
    plt.annotate("$R_{max}$", xy=(left_max_x, left_max_value), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=12, color='b')
    # 标记最低点
    plt.plot(left_min_x, left_min_value, 'bo')
    plt.annotate("$R_{min}$", xy=(left_min_x, left_min_value), xytext=(0, -16), textcoords='offset points', ha='center', fontsize=12, color='b')
    # 标记交点
    plt.plot(left_edge_x, left_edge_y, 'bo')
    plt.annotate("$R_{edge}$", xy=(left_edge_x, left_edge_y), xytext=(-24, -4), textcoords='offset points', ha='center', fontsize=12, color='b')

    plt.plot([right_max_x-right_width, right_max_x+right_width], [right_max_value, right_max_value], 'g-')
    plt.plot([right_min_x-right_width, right_min_x+right_width], [right_min_value, right_min_value], 'g-')
    # 标记最高点
    plt.plot(right_max_x, right_max_value, 'go')
    plt.annotate("$R_{max}^{'}$", xy=(right_max_x, right_max_value), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=12, color='g')
    # 标记最低点
    plt.plot(right_min_x, right_min_value, 'go')
    plt.annotate("$R_{min}^{'}$", xy=(right_min_x, right_min_value), xytext=(0, -16), textcoords='offset points', ha='center', fontsize=12, color='g')
    # 标记交点
    plt.plot(right_edge_x, right_edge_y, 'go')
    plt.annotate("$R_{edge}^{'}$", xy=(right_edge_x, right_edge_y), xytext=(24, -4), textcoords='offset points', ha='center', fontsize=12, color='g')

    # 设定标记竖线最高高度
    y_axis_min, y_axis_max = plt.ylim()
    y_axis_min -= 4
    y_axis_max += 4
    plt.ylim(y_axis_min, y_axis_max)

    left_edge_y_normalized = (left_edge_y - y_axis_min) / (y_axis_max - y_axis_min)
    right_edge_y_normalized = (right_edge_y - y_axis_min) / (y_axis_max - y_axis_min)

    plt.axvline(x=left_edge_x, ymin=0, ymax=left_edge_y_normalized, color='k', linestyle='--')
    plt.axvline(x=right_edge_x, ymin=0, ymax=right_edge_y_normalized, color='k', linestyle='--')

    # 计算测距双向箭头的起点和终点位置
    arrow_start_x = left_edge_x + 1.5
    arrow_end_x = right_edge_x - 1.5
    arrow_y = y_axis_min + 0.5 * (left_edge_y - y_axis_min)

    # 绘制双向箭头
    plt.arrow(arrow_start_x, arrow_y, arrow_end_x - arrow_start_x, 0, color='orange', width=0.03, head_width=0.5, head_length=1)
    plt.arrow(arrow_end_x, arrow_y, -(arrow_end_x - arrow_start_x), 0, color='orange', width=0.03, head_width=0.5, head_length=1)

    # 添加标注
    distance = right_edge_x - left_edge_x
    plt.annotate(f'L={distance:.1f}', xy=(left_edge_x + distance / 2, arrow_y),
                 xytext=(0, 20), textcoords='offset points', ha='center', fontsize=12, color='orange')


    print(f"L={distance:.1f}")


# 设置图像标题和轴标签
plt.title('Red Channel Analysis')
plt.xlabel('Pixel')
plt.ylabel('Red Value')

# 保存图像, 文件名为原图像文件名加上后缀'_analysis.png'
if distance is not None:
    out_path = os.path.splitext(image_path)[0] + f'_L={distance:.1f}.png'
    plt.savefig(out_path)
plt.show()
