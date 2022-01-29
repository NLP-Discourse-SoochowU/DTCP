import numpy as np


def kappa(testData, k):
    dataMat = np.mat(testData)
    P0 = 0.0
    for i in range(k):
        P0 += dataMat[i, i] * 1.0
    xsum = np.sum(dataMat, axis=1)
    ysum = np.sum(dataMat, axis=0)
    Pe = float(ysum * xsum) / k ** 2
    P0 = float(P0 / k * 1.0)
    cohens_coefficient = float((P0 - Pe) / (1 - Pe))
    return cohens_coefficient


def fleiss_kappa(testData, N, k, n):
    dataMat = np.mat(testData, float)
    oneMat = np.ones((k, 1))
    sum = 0.0
    P0 = 0.0
    for i in range(N):
        temp = 0.0
        for j in range(k):
            sum += dataMat[i, j]
            temp += 1.0 * dataMat[i, j] ** 2
        temp -= n
        temp /= (n - 1) * n
        P0 += temp
    P0 = 1.0 * P0 / N
    ysum = np.sum(dataMat, axis=0)
    for i in range(k):
        ysum[0, i] = (ysum[0, i] / sum) ** 2
    Pe = ysum * oneMat * 1.0
    ans = (P0 - Pe) / (1 - Pe)
    return ans[0, 0]


if __name__ == "__main__":
    dataArr1 = [[1.1, 1.2], [3.23, 4.78]]
    res1 = kappa(dataArr1, 2)
    print(res1)
