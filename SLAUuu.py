import numpy as np


def sqrt(x):
    if x == 0:
        return 0
    eps = 0.000001
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

    for k in range(n):
        y[k, 0] = Pb[k, 0]
        for i in range(k):
            y[k, 0] -= L[k, i] * y[i, 0]

    for k in reversed(range(n)):
        x[k, 0] = y[k, 0] / U[k, k]
        for i in range(k + 1, n):
            x[k, 0] -= U[k, i] * x[i, 0] / U[k, k]

    return x

def QR(A_, b_):
    b = Matrix(b_)
    R = Matrix(A_)
    n = R.rows
    En = Matrix.E(n,n)
    Q = En.copy()

    for i in range(n - 1):
        y = Matrix([[R[j, i]] for j in range(i, n)])
        z = Matrix([[En[j, i]] for j in range(i, n)])

        a = y[None, 0].norm(2)
        p = (y - z * a).norm(2)
        w = (y - z * a) * (1 / p)

        H = En.copy()
        H_ = (Matrix.E(n - i, n - i) - 2 * w * w.transpose())
        for j in range(i, n):
            for k in range(i, n):
                H[j, k] = H_[j - i, k - i]

        Q = Q * H.transpose()
        R = H * R

    y = Q.transpose() * b
    x = []
    for i in range(n):
        row = []
        row.append(0)
        x.append(row)
    x = Matrix(x)

    for k in reversed(range(n)):
        x[k, 0] = y[k, 0] / R[k, k]
        for i in range(k + 1, n):
            x[k, 0] = x[k, 0] - R[k, i] * x[i, 0] / R[k, k]

    return x

def MPI(A_, b_,eps):
    A = Matrix(A_)
    b = Matrix(b_)
    n = A.cols
    mu = 1 / A.norm("inf")
    B = Matrix.E(n,n) - mu * A
    B_norm = B.norm("inf")

    if B_norm >= 1:
        T = A.transpose()
        b = T * b
        A = T * A
        mu = 1 / A.norm("inf")
        B = Matrix.E(n,n) - mu * A
        B_norm = B.norm("inf")

    c = mu * b
    x = c.copy()
    k = 0
    
    while True:
        x_ = B * x + c
        k += 1
        
        if B_norm < 1 and B_norm / (1 - B_norm) * (x_ - x).norm("inf") < eps:
            return x_,k
        elif B_norm >= 1 and (A*x_ - b).norm("inf") < eps:
            return x_,k
        
        x = x_.copy()

def Zeidel(A_, b_, eps):
    A = Matrix(A_)
    b = Matrix(b_)
    n = A.cols

    if not A.is_dd():
        T = A.transpose()
        A, b = T * A, T * b

    # строим С и d
    B = Matrix.E(n,n)
    C = []
    for i in range(n):
        row = []
        row.append(0)
        C.append(row)
    C = Matrix(C)

    for i in range(n):
        C[i, 0] = b[i, 0] / A[i, i]
        for j in range(n):
            if i != j:
                B[i, j] = -1 * A[i, j] / A[i, i]
            else:
                B[i, j] = 0

    B_norm = B.norm('inf')
    x = C.copy()
    k = 0

    while True:
        x_ = x.copy()
        k += 1

        for i in range(n):
            sum_row = 0
            for j in range(n):
                if i != j:
                    sum_row += B[i, j] * x_[j, 0]
                else:
                    sum_row += C[i, 0]
            x_[i, 0] = sum_row

        if B_norm < 1 and B_norm / (1 - B_norm) * (x_ - x).norm('inf') < eps:
            return x_,k
        elif B_norm >= 1 and (A * x_ - b).norm('inf') < eps:
            return x_,k
            
        x = x_.copy()

def Jacobi(A_,b_,eps):
    A = Matrix(A_)
    b = Matrix(b_)

    if not A.is_dd():
        return
    
    n=A.rows
    B=Matrix.O(n,n)
    c=Matrix.O(n,1)
    #строим матрицу B и вектор c по методичке
    for i in range(n):
        c[i,0]=b[i,0]/A[i,i]
        for j in range(n):
            if i!=j:
                B[i,j]=-A[i,j]/A[i,i]
    
    B_norm=B.norm("inf")#норма бесконечность для критерия остановки
    
    x=c.copy()# Начальное приближение
    k=0
    while True:
        x_=B*x +c
        k+=1
          
 # Проверяем, меньше ли норма B единицы, если да, то используем апостериорную оценку, иначе смотрим невязку
        if B_norm < 1 and B_norm / (1 - B_norm) * (x_ - x).norm('inf') < eps:
            return x_,k
        elif B_norm >= 1 and (A*x_ - b).norm('inf')<eps:
            return x_,k
        x=x_.copy()

    

def test0():
    print("\nTEST 0")
    A = [[0, 2, 3], [1, 2, 4], [4, 5, 6]]
    b = [[13], [17], [32]]
    x_ = Matrix(np.linalg.solve(A, b))
    
    x1 = LU(A, b)
    print("LU")
    print(x1,(x1 - x_).norm(2))
    x2 = QR(A,b)
    print("QR")
    print(x2,(x2 - x_).norm(2))
    print("MPI")
    x3 = MPI(A,b,10**-4)
    print(x3[0],((x3[0] - x_).norm(2),x3[1]))
    print("Zeidel")
    x4 = Zeidel(A,b,10**-4)
    print(x4[0],((x4[0] - x_).norm(2),x4[1]))
    x5 = Jacobi(A, b,10**-4)
    if x5:
         print("Jacobi")
         print(x5[0],((x5[0] - x_).norm(2),x5[1]))
def test1():
    print("\nTEST 1")
    A = [[14, 1, 1], [1, 16, 1], [1, 1, 18]]
    b = [[16], [18], [20]]
    x_ = Matrix(np.linalg.solve(A, b))
    
    x1 = LU(A, b)
    print("LU")
    print(x1,(x1 - x_).norm(2))
    x2 = QR(A,b)
    print("QR")
    print(x2,(x2 - x_).norm(2))
    print("MPI")
    x3 = MPI(A,b,10**-4)
    print(x3[0],((x3[0] - x_).norm(2),x3[1]))
    print("Zeidel")
    x4 = Zeidel(A,b,10**-4)
    print(x4[0],((x4[0] - x_).norm(2),x4[1]))
    x5 = Jacobi(A, b,10**-4)
    if x5:
         print("Jacobi")
         print(x5[0],((x5[0] - x_).norm(2),x5[1]))
def test2():
    print("\nТЕСТ 2")
    A = [[-14, 1, 1], [1, -16, 1], [1, 1, -18]]
    b = [[-16], [-18], [-20]]
    x_ = Matrix(np.linalg.solve(A, b))
    
    x1 = LU(A, b)
    print("LU")
    print(x1,(x1 - x_).norm(2))
    x2 = QR(A,b)
    print("QR")
    print(x2,(x2 - x_).norm(2))
    print("MPI")
    x3 = MPI(A,b,10**-4)
    print(x3[0],((x3[0] - x_).norm(2),x3[1]))
    print("Zeidel")
    x4 = Zeidel(A,b,10**-4)
    print(x4[0],((x4[0] - x_).norm(2),x4[1]))
    x5 = Jacobi(A, b,10**-4)
    if x5:
         print("Jacobi")
         print(x5[0],((x5[0] - x_).norm(2),x5[1]))

def test3():
    print("\nТЕСТ 3")
    A=[[-14,15,16],[17,-16,13],[16,17,-18]]
    b=[[16],[18],[20]]
    x_ = Matrix(np.linalg.solve(A, b))
    
    x1 = LU(A, b)
    print("LU")
    print(x1,(x1 - x_).norm(2))
    x2 = QR(A,b)
    print("QR")
    print(x2,(x2 - x_).norm(2))
    print("MPI")
    x3 = MPI(A,b,10**-4)
    print(x3[0],((x3[0] - x_).norm(2),x3[1]))
    print("Zeidel")
    x4 = Zeidel(A,b,10**-4)
    print(x4[0],((x4[0] - x_).norm(2),x4[1]))
    x5 = Jacobi(A, b,10**-4)
    if x5:
         print("Jacobi")
         print(x5[0],((x5[0] - x_).norm(2),x5[1]))
def test4():
    print("\nТЕСТ 4")
    A=[[14,13,13],[13,16,13],[13,13,16]]
    b=[[16],[18],[20]]
    x_ = Matrix(np.linalg.solve(A, b))
    x1 = LU(A, b)
    print("LU")
    print(x1,(x1 - x_).norm(2))
    x2 = QR(A,b)
    print("QR")
    print(x2,(x2 - x_).norm(2))
    print("MPI")
    x3 = MPI(A,b,10**-4)
    print(x3[0],((x3[0] - x_).norm(2),x3[1]))
    print("Zeidel")
    x4 = Zeidel(A,b,10**-4)
    print(x4[0],((x4[0] - x_).norm(2),x4[1]))
    x5 = Jacobi(A, b,10**-4)
    if x5:
         print("Jacobi")
         print(x5[0],((x5[0] - x_).norm(2),x5[1]))
def test5(n, eps):
    
    print("ТЕСТ 5")

    A = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(1)
            if i < j:
                row.append(-1)
            if i > j:
                row.append(0)
        A.append(row)
    B = []
    for i in range(n):
        row = []
        for j in range(n):
            if i <= j:
                row.append(1)

            if i > j:
                row.append(-1)
        B.append(row)
    b = []
    for i in range(n):
        row = []
        if i == (n - 1):
            row.append(1)
        else:
            row.append(-1)
        b.append(row)

    for i in range(n):
        for j in range(n):
            A[i][j] = A[i][j] + 12*eps * B[i][j]
    x_ = Matrix(np.linalg.solve(A, b))
    
    x1 = LU(A, b)
    print("LU")
    print(x1,(x1 - x_).norm(2))
    x2 = QR(A,b)
    print("QR")
    print(x2,(x2 - x_).norm(2))
    print("MPI")
    x3 = MPI(A,b,10**-4)
    print(x3[0],((x3[0] - x_).norm(2),x3[1]))
    print("Zeidel")
    x4 = Zeidel(A,b,10**-4)
    print(x4[0],((x4[0] - x_).norm(2),x4[1]))
    x5 = Jacobi(A, b,10**-4)
    if x5:
         print("Jacobi")
         print(x5[0],((x5[0] - x_).norm(2),x5[1]))
test0()
test1()
test2()
test3()
test4()
test5(30,0.0001)



