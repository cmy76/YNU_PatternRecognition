import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#给出文件读取和存储路径
path = 'final//hw1p4_data.csv'
save_path = 'result_4//'

#读取文件为np.array
df = pd.read_csv(path)
data = np.array(df)
plt.title('original f1 vs f2 vs f3')
plt.scatter(data[:, 0], data[:, 1], c='r') #绘制f1 vs f2
# plt.savefig(save_path+'f1vsf2')
# plt.show()

plt.scatter(data[:, 1], data[:, 2], c='b') #绘制f2 vs f3
# plt.savefig('covariant//f2vsf3')
# plt.show()


plt.scatter(data[:, 0], data[:, 2], c='g')  #绘制f1 vs f3
plt.legend(['f1 vs f2', 'f2 vs f3', ' f1 vs f3']) #加入图例
plt.savefig(save_path+'original f1 vs f2 vs f3') #保存
# plt.show()
plt.close()

mean_vec0 = [np.mean(data[:, 0]), np.mean(data[:, 1]), np.mean(data[:, 2])] #求出平均向量
print('原始数据的平均向量')
print(mean_vec0)
cov0 = np.cov(data, rowvar=False) #求出协方差矩阵
print('原始数据的协方差矩阵')
print(cov0)

number = np.random.multivariate_normal(mean_vec0, cov0, 4995) #根据协方差和平均向量生成数据

# 和前面的代码一样绘制散点图
plt.title('generated f1 vs f2 vs f3')
plt.scatter(number[:, 0], number[:, 1], c='r')
# plt.savefig('covariant//n_f1vsf2')
# plt.show()

plt.scatter(number[:, 1], number[:, 2], c='b')
# plt.savefig('covariant//n_f2vsf3')
# plt.show()


plt.scatter(number[:, 0], number[:, 2], c='g')
plt.legend(['f1 vs f2', 'f2 vs f3', ' f1 vs f3'])
plt.savefig(save_path+'generated f1 vs f2 vs f3')
# plt.show()
plt.close()

mean_vec1 = [np.mean(number[:, 0]), np.mean(number[:, 1]), np.mean(number[:, 2])] #求出平均向量
print('生成数据的平均向量')
print(mean_vec1)
cov1 = np.cov(number, rowvar=False) #求出协方差矩阵
print('生成数据的协方差矩阵')
print(cov1)