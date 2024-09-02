import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_process import remove_base, muti_remove_base, fit_peaks, plot_raman

# 忽略前两行导入
path = r"C:\Users\fpc\Desktop\homework\WS2\大物实验汇报\0612_1\data.txt"
data = np.loadtxt(path)

data = data[(data[:, 0] > 50) & (data[:, 0] < 450)]
muti_remove_base(data)
fit1 = [92, 110]
fit2 = [150, 180]
fit3 = [292, 320]
fit = fit1
peaks, hfwhms, fitted_curves = fit_peaks(data, fit)

# 隔行输出peak和fwhm便于粘贴到Excel
print("Peak")
for i in range(len(peaks)):
    print(peaks[i])

print("HFWHM")
for i in range(len(hfwhms)):
    print(hfwhms[i])

plot_raman(data, fitted_curves)

plt.plot(peaks)
plt.show()
