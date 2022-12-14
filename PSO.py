import time

import numpy as np
import random
import matplotlib.pyplot as plt


# Benchmarks
def F1(X):
    output = sum(np.square(X))
    return output


def F2(X):
    output = np.sum(np.abs(X)) + np.prod(np.abs(X))
    return output


def F3(X):
    output = sum(np.square(X[j]) for i in range(len(X)) for j in range(i + 1))
    return output


def F4(X):
    output = max(np.abs(X))
    return output


def F5(X):
    output = sum(100 * np.square(X[i + 1] - np.square(X[i])) + np.square(X[i] - 1) for i in range(len(X) - 1))
    return output


def F6(X):
    output = sum(np.square(abs(i + 0.5)) for i in X)
    return output


def F7(X):
    output = sum((i + 1) * pow(X[i], 4) for i in range(len(X))) + random.random()
    return output


def F8(X):
    output = sum(-(X * np.sin(np.sqrt(np.abs(X)))))
    return output


def F9(X):
    output = 10 * len(X) + sum(pow(X[i], len(X)) - 10 * np.cos(2 * np.pi * X[i]) for i in range(len(X)))
    return output


def F10(X):
    output = -20 * np.exp(-0.2 * np.sqrt(1 / len(X) * sum(np.square(X)))) - np.exp(
        1 / len(X) * sum(np.cos(2 * np.pi * X[i]) for i in range(len(X)))) + 20 + np.exp(1)
    return output


def F11(X):
    output = 1 / 4000 * sum(np.square(X)) - np.prod(
        [np.cos(a / (b + 1) ** (1 / 2)) for a, b in zip(X, list(range(len(X))))]) + 1
    return output


def F12(X):
    a = 10
    k = 100
    m = 4
    pt1 = 0
    for i in X:
        if i > a:
            pt1 += k * pow((i - 1), m)
        elif -a <= i <= a:
            pt1 += 0
        else:
            pt1 += k * pow((-i - 1), m)
    pt2 = np.pi / len(X) * (10 * pow(np.sin(np.pi * (1 + 1 / 4 * (X[0] + 1))), 2) + sum(
        pow((1 + 1 / 4 * (X[i] + 1) - 1), 2) * (1 + 10 * pow(np.sin(np.pi * (1 + 1 / 4 * (X[i + 1] + 1))), 2)) for i in
        range(len(X) - 1)) + pow((1 + 1 / 4 * (X[len(X) - 1])), 2))
    output = pt1 + pt2
    return output


def F13(X):
    a = 5
    k = 100
    m = 4
    pt1 = 0
    for i in X:
        if i > a:
            pt1 += k * pow((i - 1), m)
        elif -a <= i <= a:
            pt1 += 0
        else:
            pt1 += k * pow((-i - 1), m)
    pt2 = 0.1 * (pow(np.sin(3 * np.pi * (X[0])), 2) + sum(
        pow(X[i] - 1, 2) * (1 + pow(np.sin(3 * np.pi * X[i] + 1), 2)) for i in range(len(X))) + pow(X[len(X) - 1],
                                                                                                    2) * (
                         1 + pow(np.sin(2 * np.pi * X[len(X) - 1]), 2)))
    return pt1 + pt2


# PSO Function
def initialize(pop, dim, lb, ub):
    X = np.zeros([pop, dim])
    V = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j]
            V[i, j] = random.random()
    return X, V


def calculateFitness(X, fun):
    fitness = fun(X)
    return fitness


def sortFitness(fitness):
    fitness = np.sort(fitness, axis=0)
    index = np.argsort(fitness, axis=0)
    return fitness, index


def sortPosition(X, V, index):
    Xnew = np.zeros(X.shape)
    Vnew = np.zeros(V.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
        Vnew[i, :] = V[index[i], :]
    return Xnew, Vnew


def updatePosition(current, currentV, Pbest, Gbest, c1, c2, w):
    newV = w * currentV + c1 * random.random() * (Pbest - current) + c2 * random.random() * (Gbest - current)
    newP = current + newV
    return newP, newV


def boundaryCheck(X, lb, ub, dim):
    for j in range(dim):
        if X[j] < lb[j]:
            X[j] = lb[j]
        elif X[j] > ub[j]:
            X[j] = ub[j]
    return X


def PSO(pop, dim, lb, ub, Max_iter, fun):
    c1 = 1  # Pbest
    c2 = 1  # Gbest
    w = 0.3  # Inertia Factor
    X, V = initialize(pop, dim, lb, ub)
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = calculateFitness(X[i, :], fun)
    fitness, sortIndex = sortFitness(fitness)
    X, V = sortPosition(X, V, sortIndex)
    GbestScore = fitness[0]
    GbestPosition = np.zeros([1, dim])
    GbestPosition[0, :] = X[0, :]
    Curve = np.zeros([Max_iter, 1])
    Xnew = np.zeros([pop, dim])
    Vnew = np.zeros([pop, dim])
    for t in range(Max_iter):
        random.seed(t + 42)
        Pbest = X[1, :]
        for i in range(pop):
            current = X[i, :]
            currentV = V[i, :]
            current, currentV = updatePosition(current, currentV, Pbest, GbestPosition, c1, c2, w)
            Xnew[i, :] = current[0]
            Vnew[i, :] = currentV[0]
            Xnew[i, :] = boundaryCheck(Xnew[i, :], lb, ub, dim)
            tempFitness = calculateFitness(Xnew[i, :], fun)
            if (tempFitness <= fitness[i]):
                fitness[i] = tempFitness
                X[i, :] = Xnew[i, :]
        Ybest, index = sortFitness(fitness)
        X, V = sortPosition(X, V, index)
        # -----------King--------------------#
        history.append(X)
        for i in range(len(fitness)):
            ave = np.average(fitness)
        fit_history.append(ave)
        print('ave=', ave)
        # -----------King--------------------#
        if (Ybest[0] <= GbestScore):
            GbestScore = Ybest[0]
            GbestPosition[0, :] = X[index[0], :]
        Curve[t] = GbestScore
        print(f"Iter {t + 1} Best: {GbestPosition} with Score: {GbestScore}")
    return Curve, GbestPosition, GbestScore


def plot_convergrnce(Curve):
    ig, ax = plt.subplots()
    ax.plot(Curve, color='dodgerblue', marker='o', markeredgecolor='dodgerblue', markerfacecolor='dodgerblue')
    ax.set_xlabel('Number of Iterations', fontsize=10)
    ax.set_ylabel('Fitness', fontsize=10)
    ax.set_xlim(-5, 200)
    ax.set_title('Convergence curve')
    plt.savefig('image.jpg', format='jpg')
    plt.show()


def plot_search_history(history):
    for i in range(len(history)):
        for n in range(len(history[0])):
            plt.scatter(history[i][n][0], history[i][n][1], c="black", alpha=0.3, facecolor='white')
            plt.xlim(lower, upper)
            plt.ylim(lower, upper)
    plt.title('Search history')
    plt.show()


def plot_fitness(fit_history):
    plt.plot(fit_history, color='b', marker='o', linewidth=2, markersize=6)
    plt.title('Average fitness of all Vultures')
    plt.xlabel('Number of Iterations', fontsize=10)
    plt.ylabel('Fitness', fontsize=10)
    plt.xlim(-5, 200)
    plt.ylim(0, 10)
    plt.show()


# Execute
def fun(X):
    return F1(X)


time_start = time.time()
pop = 500  # Population size n 1000
MaxIter = 100  # Maximum number of iterations. 500
dim = 30  # The dimension. 30
lower = -100  # The lower bound of the search interval. -100
upper = 100  # The upper bound of the search interval. 100
history = []
fit_history = []
lb = lower * np.ones([dim, 1])
ub = upper * np.ones([dim, 1])
Curve, GbestPositon, GbestScore = PSO(pop, dim, lb, ub, MaxIter, fun)
time_end = time.time()

print(f"The running time is: {time_end - time_start} s")
print('The optimal value???', GbestScore)
print('The optimal solution???', GbestPositon)
plot_convergrnce(Curve)
plot_search_history(history)
plot_fitness(fit_history)
# ---------------King---------------------#
