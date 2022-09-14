import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import time
from phe import paillier
import rsa

public_key, private_key = paillier.generate_paillier_keypair()


class LogisticSecretShare(object):

    def __init__(self, rate, lanta, Y):
        self.rate = rate
        self.size = len(Y)
        self.lanta = lanta

    def generating_matrix(self, x, y, n):
        A = np.insert(x, 0, [1], axis=0).reshape(n, 1)
        B = np.append(x, y).reshape(1, n)
        return np.dot(A, B)  # 矩阵相乘



    def paillier_(self, number_lists):
        secret_number_list = list(np.zeros((1, number_lists[0].shape[0] * number_lists[0].shape[1]))[0])
        encrypted = np.array([public_key.encrypt(x) for x in secret_number_list]).reshape(1, len(secret_number_list))

        for number_list in number_lists:
            secret_number_list = list(number_list.reshape((1, number_list.shape[1] * number_list.shape[0]))[0])
            # 加密
            encrypted += np.array([public_key.encrypt(x) for x in secret_number_list]).reshape(1,
                                                                                               len(secret_number_list))

        # 解密
        decrypted = [private_key.decrypt(x) for x in list(encrypted)[0]]
        decrypted = np.array(decrypted).reshape(number_lists[0].shape[0], number_lists[0].shape[1])

        return decrypted



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
        paillier_list = []

        for split_list in split_lists:
            Aik = np.zeros((self.n, self.n))
            for k in split_list:
                Aik += self.generating_matrix(X[k], Y[k], self.n)
            paillier_list.append(Aik)
        A = self.paillier_(paillier_list)
        return A

    def omega_zero(self, omega_init_r, z, u0):
        omega_new = (1 - 2 * self.lanta * self.rate) * omega_init_r - self.rate / self.size * (0.25 * z - 0.5 * u0)
        return omega_new

    def generate_V(self, Ar):
        V0 = Ar[0][:-1]
        V = [np.insert(V0, 0, self.size)]

        Uj = [float(Ar[0][-1])]
        for i in range(1, Ar.shape[0]):
            vj = Ar[i][:-1]
            V += [np.insert(vj, 0, Ar[0][i - 1])]
            Uj += [float(Ar[i][-1])]
        return V, Uj

    def item_model_while(self, itemmax, omega, A):
        V, Uj = self.generate_V(A)
        omega_old = omega
        for u in range(itemmax):
            omega_new = []
            for i in range(len(Uj)):
                z = np.dot(omega_old,  V[i].T)
                omega_new.append(self.omega_zero(omega_old[0][i], z, Uj[i]))
            omega_old = np.array(omega_new).T
        return omega_old


def load_csv_wisconsin(path):
    start = time.time()
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


def precision(matrix):
    return matrix[1][0]/(matrix[1][0]+(abs(matrix[0][0] - matrix[1][0])))
def recall(matrix):
    return matrix[1][0]/(matrix[1][0]+(abs(matrix[0][1] - matrix[1][1])))

# dataset = pd.read_csv(r"C:\Users\张会洋\Desktop\我的论文\australian.csv", sep='\s+', header=None) #australian
# X = dataset.iloc[:, :-1].values
# Y = dataset.iloc[:,-1].values
# # print(X.shape)
# Y = Y.tolist()
# for i in Y:
#     if i == 0:
#         Y[Y.index(i)] = -1
#     else:
#         Y[Y.index(i)] = 1
# Y = np.array(Y)


# dataset = pd.read_csv(r"C:\Users\张会洋\Desktop\sat\sat.csv", sep='\s+', header=None) #sat
# X = dataset.iloc[:, :-1].values
# Y = dataset.iloc[:,-1].values
# Y = Y.tolist()
# for i in Y:
#     if i == 1 or i == 2 or i == 3:
#         Y[Y.index(i)] = -1
#     else:
#         Y[Y.index(i)] = 1
#
# Y = np.array(Y)

# dataset = pd.read_csv(r"C:\Users\张会洋\Desktop\我的论文\processed.cleveland1.csv") #心脏病
# X = dataset.iloc[:219, :-1].values
# Y = dataset.iloc[:219, -1].values

if __name__ == '__main__':
    start = time.time()
    # path = r"C:\Users\张会洋\Desktop\我的论文\breast-cancer-wisconsin.csv"
    # X, Y = load_csv_wisconsin(pa+th)

    path = r"C:\Users\ThinkPad\Desktop\数据集\diabetes_csv.csv"
    X, Y = load_csv_diabetes(path)
    # dataset = pd.read_csv(r"C:\Users\张会洋\Desktop\我的论文\数据集\sonar.csv")  # sonar
    # X = dataset.iloc[:219, :-1].values
    # Y = dataset.iloc[:219, -1].values


    X = normalization(X)

        # from sklearn.crossvalidation import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    B = LogisticSecretShare(0.01, 0.0001, Y_train)
    A = B.split_data(X_train, Y_train, 10)
    #for m in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
    omega0 = np.random.randint(low=np.min(1), high=np.max(10), size=(1, X.shape[1]+1))
    omega1 = random.randint(0, 10) - omega0

    D = B.item_model_while(2000, omega0, A)
    end = time.time()
    print('VANE running time: %s Seconds' % (end - start))
    matrix = confusion_matrix(D[0], X_train, Y_train)
    matrix1 = confusion_matrix(D[0], X_test, Y_test)
    # print(precision(matrix), matrix)
    print('Precision =', precision(matrix1))
    print('Recall =', recall(matrix1), matrix1)





