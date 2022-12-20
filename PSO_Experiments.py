import time

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

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
def initial(pop, dim, lb, ub):
    X = np.zeros([pop, dim])
    V = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j]
            V[i, j] = random.random()
    return X, V

def CalculateFitness(X, fun, func_num):
    fitness = fun(X, func_num)
    return fitness

def SortFitness(fitness):
    fitness = np.sort(fitness, axis=0)
    index = np.argsort(fitness, axis=0)
    return fitness, index

def SortPosition(X, V, index):
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

def BorderCheck(X, lb, ub, dim):
    for j in range(dim):
        if X[j] < lb[j]:
            X[j] = lb[j]
        elif X[j] > ub[j]:
            X[j] = ub[j]
    return X

def PSO(pop, dim, lb, ub, Max_iter, fun, func_num):
    c1 = 1  # Pbest
    c2 = 1  # Gbest
    w = 0.3  # Inertia Factor
    X, V = initial(pop, dim, lb, ub)
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = CalculateFitness(X[i, :], fun, func_num)
    fitness, sortIndex = SortFitness(fitness)
    X, V = SortPosition(X, V, sortIndex)
    GbestScore = fitness[0]
    GbestPosition = np.zeros([1, dim])
    GbestPosition = X[0, :][0]
    Curve = np.zeros([Max_iter, 1])
    Xnew = np.zeros([pop, dim])
    Vnew = np.zeros([pop, dim])
    for t in range(Max_iter):
        random.seed(t+42)
        Pbest = X[1, :]
        for i in range(pop):
            current = X[i, :]
            currentV = V[i, :]
            current, currentV = updatePosition(current, currentV, Pbest, GbestPosition, c1, c2, w)
            Xnew[i, :] = current[0]
            Vnew[i, :] = currentV[0]
            Xnew[i, :] = BorderCheck(Xnew[i, :], lb, ub, dim)
            tempFitness = CalculateFitness(Xnew[i, :], fun, func_num)
            if (tempFitness <= fitness[i]):
                fitness[i] = tempFitness
                X[i, :] = Xnew[i, :]
        Ybest, index = SortFitness(fitness)
        X, V = SortPosition(X, V, index)
        # -----------King--------------------#
        history.append(X)
        for i in range(len(fitness)):
            ave = np.average(fitness)
        fit_history.append(ave)
        # print('ave=',ave)
        # -----------King--------------------#
        if (Ybest[0] <= GbestScore):
            GbestScore = Ybest[0]
            GbestPosition = X[index[0], :][0]
        Curve[t] = GbestScore
        # print(f"Iter {t+1} Best: {GbestPosition} with Score: {GbestScore}")
    return Curve, GbestPosition, GbestScore

def plot_convergrnce(Curve):
    ig, ax = plt.subplots()
    ax.plot(Curve,color='dodgerblue', marker='o', markeredgecolor='dodgerblue', markerfacecolor='dodgerblue')
    ax.set_xlabel('Number of Iterations',fontsize=10)
    ax.set_ylabel('Fitness',fontsize=10)
    ax.set_xlim(-5,200)
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
    plt.ylim(0,10)
    plt.show()

def fun(X, func_num):
    if func_num == 0:
        return F1(X)
    elif func_num == 1:
        return F2(X)
    elif func_num == 2:
        return F3(X)
    elif func_num == 3:
        return F4(X)
    elif func_num == 4:
        return F5(X)
    elif func_num == 5:
        return F6(X)
    elif func_num == 6:
        return F7(X)
    elif func_num == 7:
        return F8(X)
    elif func_num == 8:
        return F9(X)
    elif func_num == 9:
        return F10(X)
    elif func_num == 10:
        return F11(X)
    elif func_num == 11:
        return F12(X)
    elif func_num == 12:
        return F13(X)

def get_lower_upper_bound(func_num):
    if func_num == 0:
        lower = -100
        upper = 100
        return lower, upper
    elif func_num == 1:
        lower = -10
        upper = 10
        return lower, upper
    elif func_num == 2:
        lower = -100
        upper = 100
        return lower, upper
    elif func_num == 3:
        lower = -100
        upper = 100
        return lower, upper
    elif func_num == 4:
        lower = -30
        upper = 30
        return lower, upper
    elif func_num == 5:
        lower = -100
        upper = 100
        return lower, upper
    elif func_num == 6:
        lower = -128
        upper = 128
        return lower, upper
    elif func_num == 7:
        lower = -500
        upper = 500
        return lower, upper
    elif func_num == 8:
        lower = -5.12
        upper = 5.12
        return lower, upper
    elif func_num == 9:
        lower = -32
        upper = 32
        return lower, upper
    elif func_num == 10:
        lower = -600
        upper = 600
        return lower, upper
    elif func_num == 11:
        lower = -50
        upper = 50
        return lower, upper
    elif func_num == 12:
        lower = -50
        upper = 50
        return lower, upper


replication = 30

result_list =[]
runtime_list_out = []

for func_num in range(13):
    print(f"Benchmark: {func_num+1}")
    Gbest_of_all = []
    runtime_list = []
    pop = 500  # Population size n 1000
    MaxIter = 300  # Maximum number of iterations. 500
    dim = 30  # The dimension. 30
    lower, upper = get_lower_upper_bound(func_num)  # The lower and upper bound of the search interval.
    for i in range(replication):
        time_start = time.time()
        history = []
        fit_history = []
        lb = lower * np.ones([dim, 1])
        ub = upper * np.ones([dim, 1])
        Curve, GbestPositon, GbestScore = PSO(pop, dim, lb, ub, MaxIter, fun, func_num)
        time_end = time.time()
        result = fun(GbestPositon, func_num)
        Gbest_of_all.append(result)
        runtime = time_end - time_start
        print(f"【{i} round】\nThe running time is: {runtime} s")
        runtime_list.append(runtime)
        print('The optimal value：', result, "\n=================================")
    print(f'Benchmark: F {func_num + 1}\n')
    print(f"Solution Final Output: \n\tnumber of replications={replication}\n\t"
          f"Best of Gbests={np.min(Gbest_of_all)}\n\tWorst of Gbests={np.max(Gbest_of_all)}\n\taverage of Gbests={np.average(Gbest_of_all)}\n\tSTD of Gbests={np.std(Gbest_of_all)}")
    print(
        f"Runtime Final Output: \n\tnumber of replications={replication}\n\taverage of Runtime={np.average(runtime_list)}\n\t"
        f"Best of Runtime={np.min(runtime_list)}\n\tWorst of Runtime={np.max(runtime_list)}\n\tSTD of Runtime={np.std(runtime_list)}")
    temp_list_result = []
    temp_list_runtime = []
    str_func = 'F' + str(func_num + 1)
    temp_list_result.append(str_func)

    mean = np.mean(Gbest_of_all)
    standard_deviation = np.std(Gbest_of_all)
    distance_from_mean = abs(Gbest_of_all - mean)
    max_deviations = 3
    not_outlier = distance_from_mean < max_deviations * standard_deviation
    no_outliers = np.array(Gbest_of_all)[not_outlier]
    no_outliers_list = no_outliers.tolist()

    temp_list_result.append(mean)
    temp_list_result.append(np.mean(no_outliers_list))
    temp_list_result.append(np.median(Gbest_of_all))
    temp_list_result.append(np.min(Gbest_of_all))
    temp_list_result.append(np.max(Gbest_of_all))
    temp_list_result.append(standard_deviation)

    temp_list_runtime.append(str_func)
    temp_list_runtime.append(np.mean(runtime_list))
    temp_list_runtime.append(np.median(runtime_list))
    temp_list_runtime.append(np.min(runtime_list))
    temp_list_runtime.append(np.max(runtime_list))
    temp_list_runtime.append(np.std(runtime_list))

    result_list.append(temp_list_result)
    runtime_list_out.append(temp_list_runtime)

    print(Gbest_of_all)

print(np.array(result_list, dtype='object'))
df_result = pd.DataFrame(np.array(result_list, dtype='object'), columns=['Benchmark', 'Mean','Mean_no_outlier', 'Median', 'Best', 'Worst', 'STD'])
df_runtime = pd.DataFrame(np.array(runtime_list_out, dtype='object'), columns=['Benchmark', 'Mean', 'Median', 'Best', 'Worst', 'STD'])

import os
os.makedirs('Data', exist_ok=True)
df_result.to_csv('Data/pso_result_output_30.csv')
df_runtime.to_csv('Data/pso_runtime_output_30.csv')

