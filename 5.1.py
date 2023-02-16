import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import polyfit, poly1d

path = 'final//hw1p5_data.csv'  # 给定数据路径

# 读取文件为np.array
df = pd.read_csv(path)
data = np.array(df)
n_samples = len(data)

# 确定训练集
train_size = 10  # 训练集大小
degrees = range(1, 11)


def fun(coeff, x):  # 求出拟合函数的值
    val = 0
    degree = len(coeff) - 1
    for i in range(len(coeff)):
        val = val + coeff[i] * np.power(x, degree - i)
    return val


for degree in degrees:
    train_samples = np.zeros([train_size, 2])  # 为训练集预先开辟存储空间

    # 随机选择连续的10个训练数据
    low = np.random.randint(0, n_samples - train_size, size=1)
    train_samples = data[low[0]:(low[0] + train_size), :]

    # print(train_samples)
    coeff = polyfit(train_samples[:, 0], train_samples[:, 1], degree)  # 拟合
    print('系数为')
    print(coeff)
    val = np.zeros([n_samples, 1])  # 为模型的输出值预先开辟存储空间
    MSE = 0
    for i in range(n_samples):
        val[i] = fun(coeff, data[i][0])
        MSE = MSE + (val[i] - data[i][1]) ** 2
    MSE = MSE / n_samples  # 求出MSE
    print(f'MSE={MSE}')

    plt.title('y=f(x)+n')
    plt.plot(np.append(data[0:low[0], 0], data[low[0] + train_size:, 0]),  # 单独用一种颜色绘制出没有参与训练的数据
             np.append(data[0:low[0], 1], data[train_size + low[0]:, 1]))
    plt.plot(data[low[0]:low[0] + train_size, 0], data[low[0]:train_size + low[0], 1])  # 单独用一种颜色绘制出参与训练的数据
    plt.plot(data[:, 0], val)  # 绘制模型输出
    plt.legend(['f(x)+n', 'train_point', 'polyfit'])  # 加入图例
    # 存储
    path = 'result_5/result5.1and5.2// '
    path = path + 'degree = ' + str(degree) + ' train_size = ' + str(train_size) + ' MSE=' + str(MSE[0]) + '.jpg'
    plt.savefig(path)
    plt.close()
