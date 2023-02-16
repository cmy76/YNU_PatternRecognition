import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import polyfit, poly1d

path = 'final//hw1p5_data.csv'
df = pd.read_csv(path)
data = np.array(df)


def get_mse(train_size, degree, data):
    n_samples = len(data)
    train_samples = np.zeros([train_size, 2])
    low = np.random.randint(0, n_samples - train_size, size=1)
    train_samples = data[low[0]:(low[0] + train_size), :]
    coeff = polyfit(train_samples[:, 0], train_samples[:, 1], degree)
    # print(coeff)
    val = np.zeros([n_samples, 1])
    MSE = 0
    for i in range(n_samples):
        val[i] = fun(coeff, data[i][0])
        MSE = MSE + (val[i] - data[i][1]) ** 2
    MSE = MSE / n_samples
    return MSE


def fun(coeff, x): #多项式阶数
    val = 0
    degree = len(coeff) - 1
    for i in range(len(coeff)):
        val = val + coeff[i] * np.power(x, degree - i)
    return val


train_size = 10
mean_MSE_list = np.zeros([6, 1])

n = [15, 20, 25, 50, 100, 200]
degrees = range(1, 11)

# 绘制train_size-log(MSE)
for degree in degrees: #设定不同的阶数
    j = 0
    for train_size in n: #设定不同的训练尺寸
        mean_MSE = 0
        for i in range(100):
            MSE = get_mse(train_size, degree, data)
            mean_MSE = 0.01 * MSE + mean_MSE
        mean_MSE_list[j] = np.log10(mean_MSE)
        j = j+1
    save_path = 'result_5//result_5.5//train_size-log(MSE) degree=' + str(degree)
    plt.title('train_size-log(MSE) degree=' + str(degree))
    plt.stem(range(6), mean_MSE_list)
    plt.savefig(save_path)
    plt.close()

mean_MSE_list = np.zeros([10, 1])
# 绘制degree-log(MSE) 相当于第4问中的重复做(3)
for train_size in n:
    for degree in degrees:
        mean_MSE = 0
        for i in range(100):
            MSE = get_mse(train_size, degree, data)
            mean_MSE = 0.01 * MSE + mean_MSE
        mean_MSE_list[degree-1] = np.log10(mean_MSE)
    save_path = 'result_5//result_5.5//degree-log(MSE) train_size=' + str(train_size)
    plt.title('degree-log(MSE) train_size=' + str(train_size))
    plt.stem(range(1,11), mean_MSE_list)
    plt.savefig(save_path)
    plt.close()