import numpy as np
import math
import random
import matplotlib.pyplot as plt

def f(x):
    return np.tan(x) - np.cos(x) + 0.1

def unifor_n(a, b):
    n=10
    x_values=[a + (b - a) / n * i for i in range(n + 1)]
    return x_values

def optim_n(a, b):
    n=10
    x_values=[(1 / 2) * ((b - a) * np.cos(math.pi * ((2 * i + 1) / (2 * n + 2))) + b + a) for i in range(n + 1)]
    x_values.reverse()
    return x_values

def coeffs_newtone(x_values, y_values, n):
    coeff = 0

    for i in range(n):
        t = y_values[i]
        for j in range(n):  # Делим значение yi на все (xi - xj), где i - фиксированно и i!=j
            if j != i:
                t /= x_values[i] - x_values[j]
        coeff += t

    return coeff

def Newtone_poly(x, x_values, coeff):
    n = len(x_values)
    newt_f = 0
    for i in range(n):
        t = coeff[i]
        for j in range(i):
            t *= (x - x_values[j])
        newt_f += t
    return newt_f


def dNewtone_n(a, b, m):
    x_values = unifor_n(a, b)
    y_values = f(x_values)
    n=len(x_values)
    coeff=[]
    coeff.append(y_values[0])
    for i in range(2, n + 1): 
        coeff.append(coeffs_newtone(x_values[:i], y_values[:i], i))
    for i in range(m):
        trand = random.uniform(a, b)
        if i == 0:
            mx = abs(f(trand) - Newtone_poly(trand, x_values, coeff))
        else:
            mx=max(mx, abs(f(trand) - Newtone_poly(trand, x_values, coeff)))
    return mx

def dNewtone_optn(a, b, m):
    x_values = optim_n(a, b)
    y_values = f(x_values)
    n=len(x_values)
    coeff=[]
    coeff.append(y_values[0])
    for i in range(2, n + 1): 
        coeff.append(coeffs_newtone(x_values[:i], y_values[:i], i))
    for i in range(m):
        trand = random.uniform(a, b)
        if i == 0:
            mx = abs(f(trand) - Newtone_poly(trand, x_values, coeff))
        else:
            mx=max(mx, abs(f(trand) - Newtone_poly(trand, x_values, coeff)))
    return mx

def main():
    a, b = 2, 4
    print(dNewtone_n(a, b, 10000))
    print(dNewtone_optn(a, b, 10000))

    x_values = unifor_n(a, b)
    x_values_ = [a + (b - a) / 1000 * i for i in range(1000)]
    y_values = f(x_values)
    y_values_ = f(x_values_)
    n = len(x_values)
    
    coeff=[]
    coeff.append(y_values[0])
    for i in range(2, n + 1): 
        coeff.append(coeffs_newtone(x_values[:i], y_values[:i], i))
    
    y_newtone = [Newtone_poly(x_values[i], x_values, coeff) for i in range(n)]
    
    fig, ax = plt.subplots()
    ax.plot(x_values, y_newtone, linestyle='-', marker='*', linewidth='3', color='black', alpha=0.7, label='Newtone_poly')
    ax.plot(x_values_, y_values_, color=(1,0.7,0.9), label='f(x)')
    ax.set_title('Интерполяционный многочлен Ньютона', fontsize=15)
    ax.legend()
    plt.show()

main()
