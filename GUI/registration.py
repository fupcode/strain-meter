import numpy as np
import matplotlib.pyplot as plt


# 归一化
def normalize_data(data):
    # 计算每行的范数
    norms = np.linalg.norm(data, axis=1, keepdims=True)

    # 对每行进行归一化
    normalized_array = data / norms

    return normalized_array


def plot(data):
    plt.figure(figsize=(10, 5))
    for i in range(data.shape[0]):
        plt.plot(data[i, :], label=str(i))
    plt.legend()
    plt.show()


data_left = np.load("temp/left_edge_registration.npy")
data_right = np.load("temp/right_edge_registration.npy")
data = data_right - data_left
plot(data)

data = normalize_data(data)
plot(data)




