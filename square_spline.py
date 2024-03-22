import numpy as np
import math
import random
import matplotlib.pyplot as plt
#метод ньтона для квадратного корня
def sqrt(x):
    if x == 0:
        return 0
    eps = 0.00000001
    a = x / 2
    while True:
        answ = 1 / 2 * (a + x / a)
        if abs(answ - a) < eps:
            return answ
        a = answ

class Matrix:
    def __init__(self, m):
        self.data = [[m[i][j] for j in range(len(m[0]))] for i in range(len(m))]
        self.rows = len(m)
        self.cols = len(m[0])

    def __str__(self):
        matrix_str = ""
        for row in self.data:
            for element in row:
                matrix_str += str(element) + " "
            matrix_str += "\n"
        return matrix_str

    def __getitem__(self, index):
        if index[1] is None:  # [int, None]
            return Matrix([self.data[index[0]]])
        elif index[0] is None:  # [None, int]
            return Matrix([[self.data[i][index[1]]] for i in range(self.rows)])

        return self.data[index[0]][index[1]]

    def __setitem__(self, index, value):
        if index[1] is None:  # [int, None]
            for j in range(self.cols):
                self.data[index[0]][j] = value[0, j]
        elif index[0] is None:  # [None, int]
            for i in range(self.rows):
                self.data[i][index[1]] = value[i]
        else:
            self.data[index[0]][index[1]] = value

    def shape(self):
        s = []
        s.append(self.rows)
        s.append(self.cols)

        return s

    def transpose(self):
        transposed_matrix = []
        for i in range(self.cols):
            row = []
            for j in range(self.rows):
                row.append(self.data[j][i])
            transposed_matrix.append(row)
        return Matrix(transposed_matrix)

    def __add__(self, other):
        result = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                el = self.data[i][j] + other.data[i][j]
                row.append(el)
            result.append(row)
        return Matrix(result)

    def __sub__(self, other):
        return self + -1 * other

    def __mul__(self, other):
        result = []
        if isinstance(other, Matrix):
            for i in range(self.rows):
                row = []
                for j in range(other.cols):
                    el = 0
                    for k in range(self.cols):
                        el += self.data[i][k] * other.data[k][j]
                    row.append(el)
                result.append(row)
            return Matrix(result)

        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                el = self.data[i][j] * other
                row.append(el)
            result.append(row)
        return Matrix(result)

    def __rmul__(self, other):
        return self.__mul__(other)

    def norm(self, size_of_norm):
        norm = 0
        if size_of_norm == 1:
            for j in range(self.cols):
                summ = 0
                for i in range(self.rows):
                    summ += abs(self.data[i][j])
                norm = max(norm, summ)

        elif size_of_norm == "inf":
            for i in range(self.rows):
                summ = 0
                for j in range(self.cols):
                    summ += abs(self.data[i][j])
                norm = max(norm, summ)

        elif size_of_norm == 2:
            summ = 0
            for i in range(self.rows):
                for j in range(self.cols):
                    summ += abs(self.data[i][j]) ** 2
            norm = sqrt(summ)

        return norm

    def copy(self):
        result = []

        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                el = self.data[i][j]
                row.append(el)
            result.append(row)
        return Matrix(result)

    @staticmethod
    def O(rows, cols):
        return Matrix([[0 for _ in range(cols)] for _ in range(rows)])
    @staticmethod
    def E(rows, cols):
        return Matrix([[1 if i == j else 0 for i in range(cols)] for j in range(rows)])
    
    def is_dd(self):
        for i in range(self.rows):
            row_sum = 0
            for j in range(self.cols):
                row_sum = row_sum + abs(self.data[i][j]) if i != j else row_sum

            if abs(self.data[i][i]) <= row_sum:
                return False
        
        return True
#свапаем строки
def swap1(M: Matrix, max_i, i):
    temp = [0 for i in range(M.cols)]  # хранит i-ую строку
    for k in range(M.cols):
        temp[k] = (M.data[i][k])

    for k in range(M.cols):
        M[i, k] = M[max_i, k]
        M[max_i, k] = temp[k]

def LU(A, b_):
    b = Matrix(b_)
    M = Matrix(A)
    n = M.rows
    P = Matrix.E(n,n)

    #выбираем ведущие элементы и строим матрицу перестановок
    for i in range(n):

        max_el = 0
        max_i = i

        for j in range(i, n):
            if abs((M.data[j][i])) > max_el:
                max_el = abs((M.data[j][i]))
                max_i = j  # строка с ведущим элементом в i-ом столбце
        #  меняем i-ую строку с j-ой строкой в матрице М
        swap1(M, max_i, i)
        swap1(P, max_i, i)

        #  Преобразование матрицы M
        for j in range(i + 1, n):
            M[j, i] = M[j, i] / M[i, i]
            for k in range(i + 1, n):
                M[j, k] = M[j, k] - M[j, i] * M[i, k]

    # M = L + U - E
    #выполняем LU разложение
    L = Matrix.E(n,n)
    for i in range(1, n):
        for j in range(i):
            L[i, j] = M[i, j]

    U = Matrix.O(n,n)
    for i in range(n):
        for j in range(i, n):
            U[i, j] = M[i, j]

    Pb = P * b  # после перестановки
    x = Matrix.O(n, 1)
    y = Matrix.O(n, 1)

    #Ly=Pb
    #Ux=y
    for k in range(n):
        y[k, 0] = Pb[k, 0]
        for i in range(k):
            y[k, 0] -= L[k, i] * y[i, 0]
    #идем снизу вверх
    for k in reversed(range(n)):
        x[k, 0] = y[k, 0] / U[k, k]
        for i in range(k + 1, n):
            x[k, 0] -= U[k, i] * x[i, 0] / U[k, k]

    return x

def f(x):
    return np.tan(x) - np.cos(x) + 0.1

def unifor_n(a, b):
    n=2
    x_values=[a + (b - a) / n * i for i in range(n + 1)]
    return x_values

def optim_n(a, b):
    n=10
    x_values=[(1 / 2) * ((b - a) * np.cos(math.pi * ((2 * i + 1) / (2 * n + 2))) + b + a) for i in range(n + 1)]
    return x_values

def coeffs_square_spline(x_values, y_values):

    n=len(x_values)

    X=[[0 for _ in range(3*n-3)] for _ in range(3*n-3)]
    Y=[[0] for _ in range(3*n-3)]
    for i in range(n-1):
        #заполняем "блоки" матрицы коэффициентов
        X[3 * i][3 * i], X[3 * i][3 * i + 1], X[3 * i][3 * i + 2] = x_values[i] ** 2, x_values[i], 1
        X[3 * i + 1][ 3 * i], X[3 * i + 1][ 3 * i + 1], X[3 * i + 1][3 * i + 2] = x_values[i + 1] ** 2, x_values[i + 1], 1
        X[3 * i + 2][3 * i], X[3 * i + 2][3 * i + 1] = 2 * x_values[i + 1], 1
        if i != n - 2:
            X[3 * i + 2][3 * i + 3], X[3 * i + 2][ 3 * i + 4] = -2 * x_values[i + 1], -1
        Y[3 * i][0], Y[3 * i + 1][0] = y_values[i], y_values[i + 1]
    return LU(X, Y)

def square_spline(x, x_values, coef):
    n = len(x_values)
    for i in range(n - 1):
        if min(x_values[i], x_values[i + 1]) <= x <= max(x_values[i], x_values[i + 1]):
            return coef[3 * i, 0] * x ** 2 + coef[3 * i + 1, 0] * x + coef[3 * i + 2, 0]
        
def dSquare_spline_n(a, b, m):

    x_values = unifor_n(a, b)
    y_values = f(x_values)
    coeffs = coeffs_square_spline(x_values, y_values)
    for i in range(m):
        trand = random.uniform(a, b)
        if i == 0:
            mx = abs(f(trand) - square_spline(trand, x_values, coeffs))
        else:
            mx = max(mx, abs(f(trand) - square_spline(trand, x_values, coeffs)))
    return mx

def dSquare_spline_optn(a, b, m):

    x_values = unifor_n(a, b)
    y_values = f(x_values)
    coeffs = coeffs_square_spline(x_values, y_values)
    for i in range(m):
        trand = random.uniform(a, b)
        if i == 0:
            mx = abs(f(trand) - square_spline(trand, x_values, coeffs))
        else:
            mx = max(mx, abs(f(trand) - square_spline(trand, x_values, coeffs)))
    return mx

def main():
    a, b = 2, 4
    print(dSquare_spline_n(a, b, 10000))
    print(dSquare_spline_optn(a, b, 10000))

    x_values = unifor_n(a, b)
    x_values_ = [a + (b - a) / 1000 * i for i in range(1000)]
    y_values = f(x_values)
    y_values_ = f(x_values_)
    n = len(x_values)

    coeffs = coeffs_square_spline(x_values, y_values)
    y_square_spline = [square_spline(x_values[i], x_values, coeffs) for i in range(n)]

    fig, ax = plt.subplots()
    ax.plot(x_values, y_square_spline, linestyle='-', marker='*', linewidth='3', color='black', alpha=0.7, label='square_spline')
    ax.plot(x_values_, y_values_, color=(1,0.7,0.9), label='f(x)')
    ax.set_title('Квадратичный сплайн', fontsize=15)
    ax.legend()
    plt.show()

main()