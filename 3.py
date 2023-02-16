import os

import numpy as np

from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.preprocessing import scale
import cv2
import matplotlib.pyplot as plt

path = 'final//face' #指定文件路径


def generate_average_face(path: str) -> list: #生成平均脸
    images = os.listdir(path)
    X = []
    for image in images:
        if image == 'Thumbs.db': #跳过这个文件
            continue
        img_arr = cv2.imread(path + '//' + image)
        X.append(img_arr)
    n = len(X)
    # print(X)
    average_face = np.zeros([292, 240, 3]) #为平均脸声明一个存储空间
    for img in X:
        img = np.float32(img) / 255 #转为0-1的标准图像
        img = (1 / n) * img
        average_face = average_face + img
    # print(average_face)
    cv2.imshow('average_face', average_face) #显示平均脸
    # cv2.waitKey(0)
    cv2.imwrite('result_3/average_face.jpg', (average_face * 255)) #存储平均脸
    return X


def get_feature_face(X: list, n: int, n_component):
    for i in range(0, len(X)):
        X[i] = cv2.cvtColor(X[i], cv2.COLOR_BGR2GRAY)
        X[i] = X[i].flatten()
    X = scale(X) #数据标准化
    mu = np.mean(X, axis=0) #求出均值
    pca1 = PCA(n_components=n_component) #声明PCA对象
    pca1.fit(X) #计算PCA
    save_path = 'result_3//'
    for i in range(0, n):
        xhat = np.dot(pca1.transform(X[i].reshape(1, -1))[:, :50], pca1.components_[:50, :]) #矩阵乘法的方式重构图像
        xhat += mu
        img_array = np.array(xhat, dtype="int32").reshape(292, 240) #reshape为图像
        cv2.imwrite(save_path + 'feature_face' + str(i) + '.jpg', img_array * 255) #存储图像
    return X #返回标准化后的X给pca_scatter使用


def pca_scatter(X):
    mu = np.mean(X, axis=0)
    pca1 = PCA(n_components=2)
    reduced_x = pca1.fit_transform(X) #求出两维的特征
    plt.scatter([x[0] for x in reduced_x], [x[1] for x in reduced_x], c='r')
    save_path = 'result_3//'
    plt.title('PCA scatter')
    plt.savefig(save_path + 'scatter.jpg')
    # plt.show()

#调用程序
X = generate_average_face(path)
X = get_feature_face(X, 6, 50)
pca_scatter(X)
