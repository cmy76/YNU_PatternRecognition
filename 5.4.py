import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import polyfit, poly1d


def fun(coeff, x): #求出拟合函数的值(模型输出)
    val = 0
    degree = len(coeff) - 1
    for i in range(len(coeff)):
        val = val + coeff[i] * np.power(x, degree - i)
    return val


def get_result(train_size, degree, data, path): #求出MSE并绘图存储
    n_samples = len(data)
    train_samples = np.zeros([train_size, 2])
    low = np.random.randint(0, n_samples - train_size, size=1)
    train_samples = data[low[0]:(low[0] + train_size), :]
    coeff = polyfit(train_samples[:, 0], train_samples[:, 1], degree)
    val = np.zeros([n_samples, 1])
    MSE = 0
    for i in range(n_samples):
        val[i] = fun(coeff, data[i][0])
        MSE = MSE + (val[i] - data[i][1]) ** 2
    MSE = MSE / n_samples
    plt.title('y=f(x)+n')
    plt.plot(np.append(data[0:low[0], 0], data[low[0] + train_size:, 0]),
             np.append(data[0:low[0], 1], data[train_size + low[0]:, 1]))
    plt.plot(data[low[0]:low[0] + train_size, 0], data[low[0]:train_size + low[0], 1])
    plt.plot(data[:, 0], val)
    plt.legend(['f(x)+n', 'train_point', 'polyfit'])
    path = 'result_5//result_5.4//'
    path = path + 'degree = '+str(degree) + 'train_size = ' + str(train_size) + ' MSE=' + str(MSE[0]) + '.jpg'
    plt.savefig(path)
    plt.close()

#读取文件
path = 'final//hw1p5_data.csv'
df = pd.read_csv(path)
data = np.array(df)

n = [15, 20, 25, 50, 100, 200]
degrees = range(1, 11)

for train_size in n: #设定不同的训练集大小
    for degree in degrees: #设定不同的阶数，只求一次MSE,相当于重复做(1)-(2)
        get_result(train_size, degree, data, path)
