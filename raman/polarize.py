import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from data_process import remove_base, muti_remove_base, fit_peaks, plot_raman


path = r"C:\Users\fpc\Desktop\homework\WS2\极化拉曼光谱\0610\0610_raw.txt"
data = np.loadtxt(path)
data = data[(data[:, 0] > 50) & (data[:, 0] < 450)]

process_data = data.copy()

# 忽略第一列后，使第一列为第一列与最后一列的加权平均，后面依次类推
for i in range(1, data.shape[1]):
    process_data[:, i] = (data[:, i] * 2 + data[:, -i]) / 3

plot_raman(process_data, deviation=20)

# 保存名字为原名字加_processed
save_path = os.path.splitext(path)[0] + "_processed.txt"
np.savetxt(save_path, process_data)

# 查找三处峰的最大值
peak_ranges = [[90, 110], [250, 270], [300, 320]]
angles = np.arange(90, -105, -15)
peak_heights = np.zeros((len(angles), len(peak_ranges)))
for i, peak_range in enumerate(peak_ranges):
    mask = (process_data[:, 0] >= peak_range[0]) & (process_data[:, 0] <= peak_range[1])
    for j, angle in enumerate(angles):
        peak_heights[j, i] = np.max(process_data[mask, j + 1])

# 去掉最后一行后重复一份
peak_heights = peak_heights[:-1]
peak_heights = np.vstack([peak_heights, peak_heights])
angle1 = np.arange(270, -15, -15)
angle2 = np.arange(345, 270, -15)
angles = np.hstack([angle1, angle2])
peak_heights_data = np.hstack([angles.reshape(-1, 1), peak_heights])

# 保存峰高度
save_path = os.path.splitext(path)[0] + "_peak_heights.txt"
np.savetxt(save_path, peak_heights_data)
