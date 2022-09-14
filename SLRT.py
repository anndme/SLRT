import numpy as np
from sklearn.model_selection import train_test_split
import random
import time


class LogisticSecretShare(object):

    def __init__(self, rate, lanta, Y):
        self.rate = rate
        self.size = len(Y)
        self.lanta = lanta

    def generating_matrix(self, x, y, n):

        half_n = int(n / 2)
        A = x[0:half_n]
        B = x[half_n:]
        A_1 = np.hstack((A.reshape(1, A.shape[0]), np.ones((1, n - half_n))))
        A_2 = np.hstack((A.reshape(1, A.shape[0]), np.ones((1, n - half_n - 1))))
        A_2 = np.insert(A_2, 0, np.ones(1), axis=1)
        B = np.hstack((np.ones((1, half_n)), B.reshape(1, B.shape[0])))
        B_1 = np.hstack((B.reshape(1, B.shape[1]), y.reshape(1, 1)))
        # B_1 = np.array(list(B) + list(y))
        B_2 = np.insert(B, 0, [1], axis=1)
        Ma = np.dot(A_2.reshape(A_2.shape[1], A_2.shape[0]), A_1)
        Mb = np.dot(B_2.reshape(B_2.shape[1], B_2.shape[0]), B_1)
        # return np.dot(A, B)  # 矩阵相乘))
        R1 = np.random.randint(low=0, high=10, size=(n, n))
        ma_0 = Ma - R1
        ma_1 = R1
        R2 = np.random.randint(low=0, high=10, size=(n, n))
        mb_0 = Mb - R2
        mb_1 = R2
        return ma_0, ma_1, mb_0, mb_1

    def share_matrix(self, x, y, n):
        C0 = np.zeros((n, n))
        C1 = np.zeros((n, n))
        for i in range(len(x)):
            A0, A1, B0, B1 = self.generating_matrix(x[i], y[i], n)
            U = np.random.randint(low=0, high=10, size=(n, n))
            V = np.random.randint(low=0, high=10, size=(n, n))
            Z = U*V  # 第一步（1）
            # 第二步
            R = np.random.randint(low=0, high=10, size=(n, n))
            U0 = U - R
            U1 = R
            V0 = V - R
            V1 = R
            Z0 = Z - R
            Z1 = R

            # 第三步
            # S0
            E0 = A0 - U0
            F0 = B0 - V0
            # S1
            E1 = A1 - U1
            F1 = B1 - V1

            # 第四步
            E = E0 + E1
            F = F0 + F1

            # 第五步
            # SO
            C00 = U0 * F + E * V0 + Z0
            C10 = E * F + U1 * F + E * V1 + Z1
            print(np.around(C00 + C10, 3) == np.around((A0 + A1)*(B0 + B1),3))
            C0 = C00 + C0
            C1 = C10 + C1
        return C0, C1

    def omega_zero(self, omega_init_r, zr, u0):
        omega_new = (1 - 2 * self.lanta * self.rate) * omega_init_r - self.rate / self.size * (0.25 * zr - 0.5 * u0)
        return omega_new

    def secret_inner(self, a0, a1, b0, b1):

        lenth = len(b0)
        a0 = a0.reshape(1, lenth)
        a1 = a1.reshape(1, lenth)
        b0 = b0.reshape(lenth, 1)
        b1 = b1.reshape(lenth, 1)
        g = np.random.randint(low=0, high=3, size=(1, lenth))
        f = np.random.randint(low=0, high=3, size=(lenth, 1))
        h = np.dot(g, f)
        g0 = np.random.randint(low=0, high=3, size=(1, lenth))
        g1 = g - g0
        h0 = random.randint(-abs(h), abs(h))
        h1 = h - h0
        f0 = np.random.randint(low=0, high=3, size=(lenth, 1))
        f1 = f - f0
        d0 = a0 - g0
        d1 = a1 - g1
        e0 = b0 - f0
        e1 = b1 - f1
        e = e0 + e1
        d = d0 + d1
        z0 = h0 + np.dot(g0, e) + np.dot(d, f0)
        z1 = np.dot(d, e) + h1 + np.dot(g1, e) + np.dot(d, f1)
        return float(z0), float(z1)

    def generate_V(self, Ar):
        V0 = Ar[0][:-1]
        V = [np.insert(V0, 0, [self.size/2])]

        Uj = [float(Ar[0][-1])]
        for A in range(1, Ar.shape[0]):
            vj = Ar[A][:-1]
            V += [np.insert(vj, 0, Ar[0][A-1])]
            Uj += [float(Ar[A][-1])]
        return V, Uj

    def item_model_while(self, itemmax, omega0, omega1, A0, A1):
        V0, U0j = self.generate_V(A0)
        V1, U1j = self.generate_V(A1)
        omega0_old = omega0
        omega1_old = omega1
        D = []
        for u in range(itemmax):
            omega0_new = []
            omega1_new = []

            for i in range(len(U1j)):
                z0, z1 = self.secret_inner(omega0_old, omega1_old, V0[i], V1[i])

                omega0_new.append(self.omega_zero(omega0_old[0][i], z0, U0j[i]))
                omega1_new.append(self.omega_zero(omega1_old[0][i], z1, U1j[i]))

            omega0_old = np.array([omega0_new])
            omega1_old = np.array([omega1_new])
            D.append(omega0_old[0] + omega1_old[0])

        return D[-1]


def load_csv_wisconsin(path):
    X = []
    Y = []
    with open(path, 'r') as f:
        datas = f.readlines()

    for data in datas:
        split_data = data.split(',')
        d = []
        for i in split_data[1:-1]:
            try:
                d.append(float(i))
            except:
                d.append(np.NAN)
        X.append(d)
        if "2" in split_data[-1]:
            Y.append(1)
        elif "4" in split_data[-1]:
            Y.append(-1)
        else:
            Y.append(split_data[-1])
    return np.array(X), np.array(Y)


def load_csv_diabetes(path):
    X = []
    Y = []
    with open(path, 'r') as f:
        datas = f.readlines()

    for data in datas[1:]:
        split_data = data.split(',')
        X.append([float(i) for i in split_data[:-1]])
        if "positive" in split_data[-1]:  # 阳性为1，阴性为-1
            Y.append(1)
        elif "negative" in split_data[-1]:
            Y.append(-1)
        else:
            Y.append(split_data[-1])
    return np.array(X), np.array(Y)


def load_csv_australian(path):
    X = []
    Y = []
    with open(path, 'r') as f:
        datas = f.readlines()

    for data in datas:
        split_data = data.split(' ')
        d = []
        for i in split_data:
            try:
                d.append(int(float(i)))
            except:
                d.append(np.NAN)
        X.append(d)
        if "0" in split_data[-1]:
            Y.append(-1)
        elif "1" in split_data[-1]:
            Y.append(1)
        else:
            Y.append(split_data[1])
    return np.array(X), np.array(Y)

def normalization(data):  # 归一化
    max_ = data.max(axis=0)
    min_ = data.min(axis=0)
    diff = max_ - min_
    zeros = np.zeros(data.shape)
    m = data.shape[0]
    zeros = data - np.tile(min_, (m, 1))
    zeros = zeros / np.tile(diff, (m, 1))
    return zeros


def confusion_matrix(D, X, Y):
    def sigmoid(inX):
        print(inX)
        return 1.0/(1+np.exp(-inX))
    omega0 = D[0]
    omega = D[1:]
    predict = sigmoid(np.dot(omega, X.T) + omega0) > 0.5
    Y_predict = Y > 0
    result = [[np.sum(predict), len(Y) - np.sum(predict)], [np.sum(Y_predict), len(Y)-np.sum(Y_predict)]]
    return result


def accuracy(matrix):
    return (sum(matrix[1])-abs(matrix[0][0] - matrix[1][0])) / sum(matrix[1])


if __name__ == '__main__':
    # time_list = []
    # for j in [3, 50, 100]:
    #     timeit = []
    #     for i in range(100):
    path = r'C:\Users\ThinkPad\Desktop\秘密共享\数据集\australian.csv'
    X, Y = load_csv_australian(path)
    # path = r'C:\Users\ThinkPad\Desktop\秘密共享\数据集\breast-cancer-wisconsin.csv'
    # X, Y = load_csv_wisconsin(path)
    #
    # path = r"C:\Users\ThinkPad\Desktop\秘密共享\数据集\diabetes_csv.csv"
    # X, Y = load_csv_diabetes(path)
    X = normalization(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    A = LogisticSecretShare(0.01, 0.001, Y_train)
    A0, A1 = A.share_matrix(X_train, Y_train, X.shape[1] + 1)
    omega0 = np.random.randint(low=np.min(1), high=np.max(10), size=(1, X.shape[1] + 1))
    omega1 = random.randint(0, 10) - omega0
    D = A.item_model_while(2000, omega0, omega1, A0, A1)
    matrix = confusion_matrix(D, X_train, Y_train)
    matrix1 = confusion_matrix(D, X_test, Y_test)
    print(accuracy(matrix), matrix)
    print(accuracy(matrix1), matrix1)
