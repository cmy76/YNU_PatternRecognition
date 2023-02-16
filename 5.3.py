import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import polyfit, poly1d

#读取数据
path = 'final//hw1p5_data.csv'
df = pd.read_csv(path)
data = np.array(df)


#求MSE
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

#求拟合函数的值(模型输出)
def fun(coeff, x):
    val = 0
    degree = len(coeff) - 1
    for i in range(len(coeff)):
        val = val + coeff[i] * np.power(x, degree - i)
    return val


#给定训练集大小为10
train_size = 10
#预先开辟存储空间
mean_MSE_list = np.zeros([10, 1])

#固定训练集大小，计算平均MSE
for degree in range(1, 11):
    mean_MSE = 0
    for i in range(100):
        MSE = get_mse(train_size, degree, data)
        mean_MSE = 0.01 * MSE + mean_MSE
    mean_MSE_list[degree - 1] = np.log10(mean_MSE)

#绘制和存储图像
plt.title('degree-log(MSE)')
plt.stem(range(1, 11), mean_MSE_list)
save_path = 'result_5/result_5.3//'
plt.savefig(save_path + 'degree-log(MSE) train_size = 10')