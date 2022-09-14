import pandas as pd


# # dataset = pd.read_csv(r"C:\Users\张会洋\Desktop\我的论文\processed.cleveland1.csv") #心脏病
# # X = dataset.iloc[:219, :-1].values
# # Y = dataset.iloc[:219, -1].values
# # print(X.shape)
# # print(Y.shape)
# #
# # # a=[[10,10,50,50],[10,10,40,50]]
# # # b = np.tile(a,[4,1])#即向下复制4次，向右不复制
# # X = np.tile(X, [50, 1])
# # Y = np.tile(Y, [1, 50])
# # Y = np.transpose(Y)
# # print(X.shape)
# # print(Y.shape)
#
# # dataset = pd.read_csv(r"C:\Users\张会洋\Desktop\我的论文\australian.csv", sep='\s+', header=None) #ACAD
# # X = dataset.iloc[:, :-1].values
# # Y = dataset.iloc[:, -1].values
# # Y = Y.tolist()
# # for i in Y:
# #     if i == 0:
# #         Y[Y.index(i)] = -1
# #     else:
# #         Y[Y.index(i)] = 1
# # Y = np.array(Y)
# # X = np.tile(X, [20, 1])
# # Y = np.tile(Y, [1, 20])
# # Y = np.transpose(Y)
# # print(X.shape)
# # print(Y.shape)
#
#
# # def load_csv_wisconsin(path):
# #     X = []
# #     Y = []
# #     with open(path, 'r') as f:
# #         datas = f.readlines()
# #
# #     for data in datas:
# #         split_data = data.split(',')
# #         d = []
# #         for i in split_data[1:-1]:
# #             try:
# #                 d.append(float(i))
# #             except:
# #                 d.append(np.NAN)
# #         X.append(d)
# #         if "2" in split_data[-1]:
# #             Y.append(1)
# #         elif "4" in split_data[-1]:
# #             Y.append(-1)
# #         else:
# #             Y.append(split_data[-1])
# #     return np.array(X), np.array(Y)
# #
# #
# # path = r"C:\Users\张会洋\Desktop\我的论文\breast-cancer-wisconsin.csv"
# # X, Y = load_csv_wisconsin(path)
# # X = np.tile(X, [20, 1])
# # Y = np.tile(Y, [1, 20])
# # Y = np.transpose(Y)
# # print(X.shape)
# # print(Y.shape)
#
#
#
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
        A = np.insert(x, 0, [1], axis=0).reshape(n, 1)
        B = np.append(x, y).reshape(1, n)
        return np.dot(A, B)  # 矩阵相乘

    def split_data_half(self, X, Y):
        self.n = X.shape[1] + 1  # 确定维数
        Aik_0 = np.zeros((self.n, self.n))
        Aik_1 = np.zeros((self.n, self.n))
        for i in range(len(Y)):
            Aik = self.generating_matrix(X[i], Y[i], self.n)
            R = np.random.randint(low=0, high=10, size=(self.n, self.n))
            Aik_0 += R
            Aik_1 += Aik - R
        return Aik_0, Aik_1

    def split_data(self, X, Y, N):

        self.n = X.shape[1] + 1  # 确定维数
        chioce = [i for i in range(len(Y))]
        size = int(len(Y) / N)
        split_lists = []
        for i in range(N-1):
            data = []
            for j in range(size):
                data += [chioce.pop(random.randint(0, len(chioce)-1))]
            split_lists.append(data)
        split_lists.append(chioce)
        Aik_0 = np.zeros((self.n, self.n))
        Aik_1 = np.zeros((self.n, self.n))
        for split_list in split_lists:
            Aik_0_, Aik_1_ = self.split_data_half(X[split_list], Y[split_list])
            Aik_0 += Aik_0_
            Aik_1 += Aik_1_

        return Aik_0, Aik_1

    def omega_zero(self, omega_init_r, zr, u0):
        omega_new = (1 - 2 * self.lanta * self.rate) * omega_init_r - self.rate / self.size * (0.25 * zr - 0.5 * u0)
        return omega_new

    def secret_inner(self, a0, a1, b0, b1):

        lenth = len(b0)
        a0 = a0.reshape(1, lenth)
        a1 = a1.reshape(1, lenth)
        b0 = b0.reshape(lenth, 1)
        b1 = b1.reshape(lenth, 1)
        g = np.random.randint(low=0, high=10, size=(1, lenth))
        f = np.random.randint(low=0, high=10, size=(lenth, 1))
        h = np.dot(g, f)
        g0 = np.random.randint(low=0, high=10, size=(1, lenth))
        g1 = g - g0
        h0 = random.randint(-abs(h), abs(h))
        h1 = h - h0
        f0 = np.random.randint(low=0, high=10, size=(lenth, 1))
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
        return 1.0/(1+np.exp(-inX))
    omega0 = D[0]
    omega = D[1:]
    predict = sigmoid(np.dot(omega, X.T) + omega0) > 0.5
    Y_predict = Y > 0
    result = [[np.sum(predict), len(Y) - np.sum(predict)], [np.sum(Y_predict), len(Y)-np.sum(Y_predict)]]
    return result


def accuracy(matrix):
    return (sum(matrix[1])-abs(matrix[0][0] - matrix[1][0])) / sum(matrix[1])


dataset = pd.read_csv(r"C:\Users\张会洋\Desktop\我的论文\processed.cleveland1.csv") #心脏病
X = dataset.iloc[:219, :-1].values
Y = dataset.iloc[:219, -1].values
# a=[[10,10,50,50],[10,10,40,50]]
# b = np.tile(a,[4,1])#即向下复制4次，向右不复制
C = np.tile(X, [50, 1])
B = np.tile(Y, [1, 50])
B = np.transpose(B)


#
# dataset = pd.read_csv(r"C:\Users\张会洋\Desktop\我的论文\australian.csv", sep='\s+', header=None) #ACAD
# X = dataset.iloc[:, :-1].values
# Y = dataset.iloc[:, -1].values
# Y = Y.tolist()
# for i in Y:
#     if i == 0:
#         Y[Y.index(i)] = -1
#     else:
#         Y[Y.index(i)] = 1
# Y = np.array(Y)
# C = np.tile(X, [20, 1])
# B = np.tile(Y, [1, 20])
# B = np.transpose(B)


if __name__ == '__main__':

    # time_list = []
    # for j in [3, 50, 100]:
    #     timeit = []
    #     for i in range(100):
    # path = r"C:\Users\张会洋\Desktop\我的论文\breast-cancer-wisconsin.csv"
    # X, Y = load_csv_wisconsin(path)
    # C = np.tile(X, [20, 1])
    # B = np.tile(Y, [1, 20])
    # B = np.transpose(B)

    # path = r"C:\Users\张会洋\Desktop\我的论文\diabetes_csv.csv"
    # X, Y = load_csv_diabetes(path)
    # C = np.tile(X, [20, 1])
    # B = np.tile(Y, [1, 20])
    # B = np.transpose(B)
    for i in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
        start = time.time()
        X = C[0:i]
        Y = B[0:i]
        X = normalization(X)
                # from sklearn.cross_validation import train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
        A = LogisticSecretShare(0.01, 0.0001, Y_train)
        A0, A1 = A.split_data(X_train, Y_train, 10)

        #         timeit.append(end-start)
        #     print(j, sum(timeit)/len(timeit))
        #     time_list.append(sum(timeit)/len(timeit))
        # print(time_list)


        omega0 = np.random.randint(low=np.min(1), high=np.max(10), size=(1,X.shape[1]+1))
        omega1 = random.randint(0, 10) - omega0
        D = A.item_model_while(1000, omega0, omega1, A0, A1)
        end = time.time()
        print('%s,' % (end - start))


    matrix = confusion_matrix(D, X_train, Y_train)
    matrix1 = confusion_matrix(D, X_test, Y_test)
    print(accuracy(matrix), matrix)
    print(accuracy(matrix1), matrix1)


# # #同态加密
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# import random
# import time
# from phe import paillier
#
# public_key, private_key = paillier.generate_paillier_keypair()
#
#
# class LogisticSecretShare(object):
#
#     def __init__(self, rate, lanta, Y):
#         self.rate = rate
#         self.size = len(Y)
#         self.lanta = lanta
#
#     def generating_matrix(self, x, y, n):
#         A = np.insert(x, 0, [1], axis=0).reshape(n, 1)
#         B = np.append(x, y).reshape(1, n)
#         return np.dot(A, B)  # 矩阵相乘
#
#
#
#     def paillier_(self, number_lists):
#         secret_number_list = list(np.zeros((1, number_lists[0].shape[0] * number_lists[0].shape[1]))[0])
#         encrypted = np.array([public_key.encrypt(x) for x in secret_number_list]).reshape(1, len(secret_number_list))
#
#         for number_list in number_lists:
#             secret_number_list = list(number_list.reshape((1, number_list.shape[1] * number_list.shape[0]))[0])
#             # 加密
#             encrypted += np.array([public_key.encrypt(x) for x in secret_number_list]).reshape(1,
#                                                                                                len(secret_number_list))
#
#         # 解密
#         decrypted = [private_key.decrypt(x) for x in list(encrypted)[0]]
#         decrypted = np.array(decrypted).reshape(number_lists[0].shape[0], number_lists[0].shape[1])
#
#         return decrypted
#
#     start = time.time()
#
#     def split_data(self, X, Y, N):
#
#         self.n = X.shape[1] + 1  # 确定维数
#         chioce = [i for i in range(len(Y))]
#         size = int(len(Y) / N)
#         split_lists = []
#         for i in range(N-1):
#             data = []
#             for j in range(size):
#                 data += [chioce.pop(random.randint(0, len(chioce)-1))]
#             split_lists.append(data)
#         split_lists.append(chioce)
#         paillier_list = []
#
#         for split_list in split_lists:
#             Aik = np.zeros((self.n, self.n))
#             for k in split_list:
#                 Aik += self.generating_matrix(X[k], Y[k], self.n)
#             paillier_list.append(Aik)
#         A = self.paillier_(paillier_list)
#         return A
#
#     def omega_zero(self, omega_init_r, z, u0):
#         omega_new = (1 - 2 * self.lanta * self.rate) * omega_init_r - self.rate / self.size * (0.25 * z - 0.5 * u0)
#         return omega_new
#
#     def generate_V(self, Ar):
#         V0 = Ar[0][:-1]
#         V = [np.insert(V0, 0, self.size)]
#
#         Uj = [float(Ar[0][-1])]
#         for i in range(1, Ar.shape[0]):
#             vj = Ar[i][:-1]
#             V += [np.insert(vj, 0, Ar[0][i - 1])]
#             Uj += [float(Ar[i][-1])]
#         return V, Uj
#
#     def item_model_while(self, itemmax, omega, A):
#         V, Uj = self.generate_V(A)
#         omega_old = omega
#         for u in range(itemmax):
#             omega_new = []
#             for i in range(len(Uj)):
#                 z = np.dot(omega_old,  V[i].T)
#                 omega_new.append(self.omega_zero(omega_old[0][i], z, Uj[i]))
#             omega_old = np.array(omega_new).T
#         return omega_old
#
#
# def load_csv_wisconsin(path):
#     X = []
#     Y = []
#     with open(path, 'r') as f:
#         datas = f.readlines()
#
#     for data in datas:
#         split_data = data.split(',')
#         d = []
#         for i in split_data[1:-1]:
#             try:
#                 d.append(float(i))
#             except:
#                 d.append(np.NAN)
#         X.append(d)
#         if "2" in split_data[-1]:
#             Y.append(1)
#         elif "4" in split_data[-1]:
#             Y.append(-1)
#         else:
#             Y.append(split_data[-1])
#     return np.array(X), np.array(Y)
#
#
# def load_csv_diabetes(path):
#     X = []
#     Y = []
#     with open(path, 'r') as f:
#         datas = f.readlines()
#
#     for data in datas[1:]:
#         split_data = data.split(',')
#         X.append([float(i) for i in split_data[:-1]])
#         if "positive" in split_data[-1]:  # 阳性为1，阴性为-1
#             Y.append(1)
#         elif "negative" in split_data[-1]:
#             Y.append(-1)
#         else:
#             Y.append(split_data[-1])
#     return np.array(X), np.array(Y)
#
#
# def normalization(data):  # 归一化
#     max_ = data.max(axis=0)
#     min_ = data.min(axis=0)
#     diff = max_ - min_
#     zeros = np.zeros(data.shape)
#     m = data.shape[0]
#     zeros = data - np.tile(min_, (m, 1))
#     zeros = zeros / np.tile(diff, (m, 1))
#     return zeros
#
#
# def confusion_matrix(D, X, Y):
#     def sigmoid(inX):
#         return 1.0/(1+np.exp(-inX))
#     omega0 = D[0]
#     omega = D[1:]
#     predict = sigmoid(np.dot(omega, X.T) + omega0) > 0.5
#     Y_predict = Y > 0
#     result = [[np.sum(predict), len(Y) - np.sum(predict)], [np.sum(Y_predict), len(Y)-np.sum(Y_predict)]]
#     return result
#
#
# def accuracy(matrix):
#     return (sum(matrix[1])-abs(matrix[0][0] - matrix[1][0])) / sum(matrix[1])
#
# dataset = pd.read_csv(r"C:\Users\张会洋\Desktop\我的论文\processed.cleveland1.csv") #心脏病
# X = dataset.iloc[:219, :-1].values
# Y = dataset.iloc[:219, -1].values
# # a=[[10,10,50,50],[10,10,40,50]]
# # b = np.tile(a,[4,1])#即向下复制4次，向右不复制
# M = np.tile(X, [20, 1])
# N = np.tile(Y, [1, 20])
# N = np.transpose(N)
#
#
#
# # dataset = pd.read_csv(r"C:\Users\张会洋\Desktop\我的论文\australian.csv", sep='\s+', header=None) #ACAD
# # X = dataset.iloc[:, :-1].values
# # Y = dataset.iloc[:, -1].values
# # Y = Y.tolist()
# # for i in Y:
# #     if i == 0:
# #         Y[Y.index(i)] = -1
# #     else:
# #         Y[Y.index(i)] = 1
# # Y = np.array(Y)
# # M = np.tile(X, [20, 1])
# # N = np.tile(Y, [1, 20])
# # N = np.transpose(N)
#
# if __name__ == '__main__':
#     # start = time.time()
#         # time_list = []
#         # for j in [3, 50, 100]:
#         #     timeit = []
#         #     for i in range(100):
#         # path = r"C:\Users\张会洋\Desktop\我的论文\breast-cancer-wisconsin.csv"
#         # X, Y = load_csv_wisconsin(path)
#         # C = np.tile(X, [20, 1])
#         # B = np.tile(Y, [1, 20])
#         # B = np.transpose(B)
#     # path = r"C:\Users\张会洋\Desktop\我的论文\breast-cancer-wisconsin.csv"
#     # X, Y = load_csv_wisconsin(path)
#     # M = np.tile(X, [20, 1])
#     # N = np.tile(Y, [1, 20])
#     # N = np.transpose(N)
#     # for i in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
#     #     start = time.time()
#     #     X = M[0:i]
#     #     Y = N[0:i]
#
#     # path = r"C:\Users\张会洋\Desktop\我的论文\diabetes_csv.csv"
#     # X, Y = load_csv_diabetes(path)
#     # M = np.tile(X, [20, 1])
#     # N = np.tile(Y, [1, 20])
#     # N = np.transpose(N)
#     for i in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
#         start = time.time()
#         X = M[0:i]
#         Y = N[0:i]
#         X = normalization(X)
#         X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
#         B = LogisticSecretShare(0.01, 0.0001, Y_train)
#         A = B.split_data(X_train, Y_train, 10)
#         end = time.time()
#         print('%s, ' % (end - start))
#         # for m in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
#         omega0 = np.random.randint(low=np.min(1), high=np.max(10), size=(1, X.shape[1] + 1))
#         omega1 = random.randint(0, 10) - omega0
#
#         D = B.item_model_while(1000, omega0, A)
#
#     matrix = confusion_matrix(D[0], X_train, Y_train)
#     matrix1 = confusion_matrix(D[0], X_test, Y_test)
#     print(accuracy(matrix), matrix)
#     print(accuracy(matrix1), matrix1)










