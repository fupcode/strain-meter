from matplotlib import rcParams
import matplotlib.pyplot as plt

fontsize = 12
rcParams['font.family'] = 'Arial'
rcParams['font.serif'] = 'Times New Roman'
rcParams['font.size'] = fontsize
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

def plot_trajectory(x, y, title='Trajectory', xlabel='X', ylabel='Y', data_marker_size=10, start_end_marker_size=20):
    plt.plot(x, y, 'b-', lw=1)  # 使用蓝色的线条
    plt.scatter(x, y, c='orange', s=data_marker_size)  # 使用橙色的数据点
    plt.scatter(x[0], y[0], c='r', label='Start', s=start_end_marker_size)  # 使用红色表示起点
    plt.scatter(x[-1], y[-1], c='g', label='End', s=start_end_marker_size)  # 使用绿色表示终点
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.axis('equal')
    plt.legend()  # 添加图例
    plt.show()
