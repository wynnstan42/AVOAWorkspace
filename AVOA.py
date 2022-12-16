import math
import random
import time

import numpy as np


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


# AVOA
def initializeVultures(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j]
    return X


def caculateFitness(X, func):
    fitness = func(X)
    return fitness


def sortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index


def sortPosition(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew


def boundaryCheck(X, lb, ub, dim):
    for j in range(dim):
        if X[j] < lb[j]:
            X[j] = ub[j]
        elif X[j] > ub[j]:
            X[j] = lb[j]
    return X


def rouletteWheelSelection(first_best_vulture, second_best_vulture, L1, L2):
    # equation(1)
    probability = [L1, L2]
    index = sum(np.where(random.random() <= np.cumsum([probability])))
    if index.all() > 0:
        target_vulture = first_best_vulture
    else:
        target_vulture = second_best_vulture
    return target_vulture


def explorationPhase(current_vulture, target_vulture, F, p1, upper_bound, lower_bound):
    if p1 > random.random():  # Behavior 1
        current_vulture = target_vulture - (abs((2 * random.random()) * target_vulture - current_vulture)) * F
    else:  # Behavior 2
        current_vulture = (target_vulture - F + random.random() * (
                (upper_bound - lower_bound) * random.random() + lower_bound))
    return current_vulture


def exploitationPhase(current_vulture, first_best_vulture, second_best_vulture, target_vulture, F, p2, p3, dim):

    if abs(F) >= 0.5:  # Exploitation Phase I

        if p2 >= random.random():  # Behavior 3
            # equation(10)
            current_vulture = (abs((2 * random.random()) * target_vulture - current_vulture)) * (
                    F + random.random()) - (target_vulture - current_vulture)

        else:  # Behavior 4
            # equation(13)
            s1 = target_vulture * (random.random() * current_vulture / (2 * np.pi)) * np.cos(current_vulture)
            s2 = target_vulture * (random.random() * current_vulture / (2 * np.pi)) * np.sin(current_vulture)
            current_vulture = target_vulture - (s1 + s2)

    else:  # Exploitation Phase II

        if p3 >= random.random():  # Behavior 5
            # equation(16)
            # A1 = first_best_vulture - ((np.multiply(first_best_vulture, current_vulture)) / (first_best_vulture - current_vulture ** 2)) * F
            A1 = first_best_vulture - ((first_best_vulture * current_vulture) / (first_best_vulture - current_vulture ** 2)) * F
            A2 = second_best_vulture - ((second_best_vulture * current_vulture) / (second_best_vulture - current_vulture ** 2)) * F
            current_vulture = (A1 + A2) / 2

        else:  # Behavior 6
            current_vulture = target_vulture - abs(target_vulture - current_vulture) * F * levyFlight(dim)

    return current_vulture

def levyFlight(dim):  # equation(18)
    beta = 1.5
    sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(1, dim)
    v = np.random.randn(1, dim)
    LF = u * sigma / abs(v) ** (1 / beta)
    return LF


def AVOA(pop, dim, lb, ub, max_iterations, func):

    # Set Params
    L1 = 0.8
    L2 = (1 - L1)
    p1 = 0.6
    p2 = 0.4
    p3 = 0.6
    Gama = 2.5

    # 【Initialize】
    X = initializeVultures(pop, dim, lb, ub)  # Initialize random vultures
    fitness = np.zeros([pop, 1])

    # 【Evaluate Vultures】*for the initial vultures
    for i in range(pop):
        fitness[i] = caculateFitness(X[i, :], func)
    fitness, sortIndex = sortFitness(fitness)  # Sort the fitness values of African Vultures
    X = sortPosition(X, sortIndex)  # Sort the African Vultures population based on fitness
    gBestFitness = fitness[0]  # Stores the optimal value for the current iteration.
    gBestPositon = X[0, :][0]
    Xnew = np.zeros([pop, dim])

    # Start Iterating
    for current_iter in range(max_iterations):
        # 【Find Best Vultures】
        first_best_vulture = X[0, :]  # First Best Vulture
        second_best_vulture = X[1, :]  # Second Best Vulture

        # loop each vulture
        for i in range(pop):
            current_vulture_X = X[i, :]

            # 【Select Target Vulture】
            target_vulture = rouletteWheelSelection(first_best_vulture, second_best_vulture, L1, L2)

            # 【Calculate Starvation Rate F】
            # equation(3)
            t = np.random.uniform(-2, 2) * (
                    (np.sin((np.pi / 2) * (current_iter / max_iterations)) ** Gama)
                    + np.cos((np.pi / 2) * (current_iter / max_iterations))
                    - 1)
            # equation(4)
            F = (2 * random.random() + 1) * np.random.uniform(-1, 1) * (1 - (current_iter / max_iterations)) + t

            # 【Update Position】
            if abs(F) >= 1:  # Exploration Phase (|F|>1)
                current_vulture_X = explorationPhase(current_vulture_X, target_vulture, F, p1, ub, lb)

            else:  # Exploitation Phase (|F|<=1)
                current_vulture_X = exploitationPhase(current_vulture_X, first_best_vulture, second_best_vulture, target_vulture, F, p2, p3, dim)

            # Evaluate new position
            Xnew[i, :] = current_vulture_X[0]
            Xnew[i, :] = boundaryCheck(Xnew[i, :], lb, ub, dim)
            tempFitness = caculateFitness(Xnew[i, :], func)

            # Update personal best
            if tempFitness <= fitness[i]:
                fitness[i] = tempFitness
                X[i, :] = Xnew[i, :]

        # Update global best
        yBest, index = sortFitness(fitness)
        X = sortPosition(X, index)
        if yBest[0] <= gBestFitness:
            gBestFitness = yBest[0]
            gBestPositon = X[index[0], :][0]

    return gBestPositon, gBestFitness


def func(X):
    return F2(X)


pop = 30  # Population size
MaxIter = 500  # Maximum number of iterations
dim = 30  # Dimensions
lower_bound = -100  # Lower bound of the benchmark
upper_bound = 100  # Upper bound of the benchmark
lower_bounds = lower_bound * np.ones([dim, 1])
upper_bounds = upper_bound * np.ones([dim, 1])

time_start = time.time()
gBestPositon, gBestScore = AVOA(pop, dim, lower_bounds, upper_bounds, MaxIter, func)  # AVOA
time_end = time.time()
print(f"The running time is: {time_end - time_start} s")
print('The optimal value：', gBestScore)
print('The optimal solution：', gBestPositon)
