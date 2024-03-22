import numpy as np
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
    n = 10
    x_values = [a + (b - a) / n * i for i in range(n + 1)]
    return x_values

def optim_n(a, b):
    n = 10
    x_values = [(1 / 2) * ((b - a) * np.cos(np.pi * ((2 * i + 1) / (2 * n + 2))) + b + a) for i in range(n + 1)]
    return x_values

def coeffs_cubic_spline(x_values, y_values):
    """Функция поиска коэффициентов кубического сплайна"""
    n = len(x_values)
    x_increment = [x_values[i + 1] - x_values[i] for i in range(n - 1)]  # построим приращения h
    y_increment = [y_values[i + 1] - y_values[i] for i in range(n - 1)]  # построим приращение y, чтобы меньше писать было


    H = [[0 for _ in range(n - 2)] for _ in range(n - 2)]
    gamma = [[0] for _ in range(n - 2)]
    for i in range(n - 2):
        H[i][i] = 2 * (x_increment[i + 1] + x_increment[i])
        if i > 0:
            H[i][i - 1] = x_increment[i]
            H[i - 1][i] = x_increment[i]

        gamma[i][0] = 6 * (y_increment[i + 1] / x_increment[i + 1] - y_increment[i] / x_increment[i])

    solve = LU(H, gamma)
    y_values__ = [0] + [solve[i, 0] for i in range(n - 2)] + [0] # Вторые производные

    y_values_ = []  # Первые производные
    for i in range(n - 1):
        y_values_.append(y_increment[i] / x_increment[i] - y_values__[i + 1] * x_increment[i] / 6 - y_values__[i] * x_increment[i] / 3)

    coeffs = [[0] for _ in range(4*n-4)]
    for i in range(n - 1):
        coeffs[4 * i][0] = y_values[i]
        coeffs[4 * i + 1][0] = y_values_[i]
        coeffs[4 * i + 2][0] = y_values__[i] / 2
        coeffs[4 * i + 3][0] = (y_values__[i + 1] - y_values__[i]) / (6 * x_increment[i])

    return coeffs

def cubic_spline(x, x_values, coeffs):
    
    n = len(x_values)

    for i in range(n - 1):
        if min(x_values[i], x_values[i + 1]) <= x <= max(x_values[i], x_values[i + 1]):
            return coeffs[4 * i][0] + coeffs[4 * i + 1][0] * (x - x_values[i]) + \
                   coeffs[4 * i + 2][0] * (x - x_values[i]) ** 2 + coeffs[4 * i + 3][0] * (x - x_values[i]) ** 3

def dCubic_spline_n(a, b, m):

    x_values = unifor_n(a, b)
    y_values = f(x_values)
    coeffs = coeffs_cubic_spline(x_values, y_values)
    for i in range(m):
        trand = random.uniform(a, b)
        if i == 0:
            mx = abs(f(trand) - cubic_spline(trand, x_values, coeffs))
        else:
            mx = max(mx, abs(f(trand) - cubic_spline(trand, x_values, coeffs)))
    return mx

def dCubic_spline_optn(a, b, m):

    x_values = unifor_n(a, b)
    y_values = f(x_values)
    coeffs = coeffs_cubic_spline(x_values, y_values)
    for i in range(m):
        trand = random.uniform(a, b)
        if i == 0:
            mx = abs(f(trand) - cubic_spline(trand, x_values, coeffs))
        else:
            mx = max(mx, abs(f(trand) - cubic_spline(trand, x_values, coeffs)))
    return mx


def main():
    a, b = 2, 4
    x = 2.754
    print(dCubic_spline_n(a, b, 10000))
    print(dCubic_spline_optn(a, b, 10000))

    x_values = unifor_n(a, b)
    x_values_ = [a + (b - a) / 1000 * i for i in range(1000)]
    y_values = f(x_values)
    y_values_ = f(x_values_)
    n = len(x_values)

    coeffs = coeffs_cubic_spline(x_values, y_values)
    y_cubic_spline = [cubic_spline(x_values[i], x_values, coeffs) for i in range(n)]

    fig, ax = plt.subplots()
    ax.plot(x_values, y_cubic_spline, linestyle='-', marker='*', linewidth='3', color='black', alpha=0.7, label='cubic_spline')
    ax.plot(x_values_, y_values_, color=(1, 0.7, 0.9), label='f(x)')
    ax.set_title('Кубический сплайн', fontsize=15)
    ax.legend()
    plt.show()

main()