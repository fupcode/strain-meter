import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import peakutils


def remove_base(intensity):
    base = peakutils.baseline(intensity)  # 背底拟合
    intensity_r = intensity - base  # 扣除背底
    return intensity_r


def muti_remove_base(data):
    for i in range(1, data.shape[1]):
        data[:, i] = remove_base(data[:, i])


def gaussian(x, y0, amplitude, mean, stddev):
    """定义带基线偏移的高斯函数"""
    return y0 + amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))


def fit_peaks(data, fit_range):
    """
    对数据中的每一列强度进行高斯拟合，并返回峰中心、半高宽和拟合曲线

    参数：
    data: numpy数组，第一列为拉曼峰移，后面几列为不同时期测得的强度
    fit_range: 拟合的局部范围，例如 [140, 160]

    返回：
    peaks: 峰中心数组
    hfwhms: 半高宽数组
    fitted_curves: 拟合曲线列表
    """
    x = data[:, 0]  # 拉曼峰移
    y_data = data[:, 1:]  # 各时期测得的强度
    peaks = []
    hfwhms = []
    fitted_curves = []

    # 选择拟合的局部范围
    mask = (x >= fit_range[0]) & (x <= fit_range[1])
    x_local = x[mask]

    for y in y_data.T:
        y_local = y[mask]
        initial_guess = [min(y_local), max(y_local) - min(y_local), np.mean(x_local), 10]
        bounds = ([-100, 0, fit_range[0], 0], [100, np.inf, fit_range[1], np.inf])
        try:
            popt, _ = curve_fit(gaussian, x_local, y_local, p0=initial_guess, bounds=bounds)
            y0, amplitude, mean, stddev = popt
            hfwhm = np.sqrt(2 * np.log(2)) * stddev  # 计算半高宽
            peaks.append(mean)
            hfwhms.append(hfwhm)
            fitted_curve = gaussian(x, *popt)
            fitted_curves.append(fitted_curve)
        except RuntimeError:
            # 如果拟合失败，返回 NaN
            peaks.append(np.nan)
            hfwhms.append(np.nan)
            fitted_curves.append(np.full_like(x, np.nan))

    return np.array(peaks), np.array(hfwhms), np.array(fitted_curves).T


def plot_raman(data, fitted_curves=None, plot_range=None, deviation=50):
    """
    绘制拉曼光谱图，每个数据向上平移以便于观察，如果提供拟合曲线则同时绘制为虚线

    参数：
    data: numpy数组，第一列为拉曼峰移，后面几列为不同时期测得的强度
    fitted_curves: 拟合曲线列表
    """
    x = data[:, 0]
    y_data = data[:, 1:]
    for i, y in enumerate(y_data.T):
        plt.plot(x, y + i * deviation)

    if fitted_curves is not None:
        for i, fitted_curve in enumerate(fitted_curves.T):
            plt.plot(x, fitted_curve + i * deviation, '--', color='black')

    if plot_range is not None:
        plt.xlim(plot_range)
    plt.xlabel("Raman shift (cm-1)")
    plt.ylabel("Intensity")
    plt.show()
