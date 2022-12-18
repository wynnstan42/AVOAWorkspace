import math
import random
import time
import numpy as np
import pandas as pd
import os
import sys
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

class Student:
    def __init__(self, fitness, func_num, dim, minx, maxx, seed):
        self.rnd = random.Random(seed)

        # a list of size dim
        # with 0.0 as value of all the elements
        self.position = [0.0 for i in range(dim)]

        # loop dim times and randomly select value of decision var
        # value should be in between minx and maxx
        for i in range(dim):
            self.position[i] = ((maxx - minx) *
                                self.rnd.random() + minx)
        self.func_num = func_num

        # compute the fitness of student
        self.fitness = fitness(self.position, self.func_num)




# Teaching learning based optimization
def tlbo(fitness, func_num, max_iter, n, dim, minx, maxx, conver, history, fit_history):
    rnd = random.Random(0)

    # create n random students
    classroom = [Student(fitness, func_num, dim, minx, maxx, i) for i in range(n)]

    # compute the value of best_position and best_fitness in the classroom
    Xbest = [0.0 for i in range(dim)]
    Fbest = sys.float_info.max

    for i in range(n):  # check each Student
        if classroom[i].fitness < Fbest:
            Fbest = classroom[i].fitness
            Xbest = classroom[i].position.copy()

    # main loop of tlbo
    Iter = 0
    while Iter < max_iter:
        # after every 10 iterations
        # print iteration number and best fitness value so far
        # if Iter % 10 == 0 and Iter > 1:
        #     print("Iter = " + str(Iter) + " best fitness = %f" % Fbest)

        # for each student of classroom
        for i in range(n):

            ### Teaching phase of ith student

            # compute the mean of all the students in the class
            Xmean = [0.0 for i in range(dim)]
            for k in range(n):
                for j in range(dim):
                    Xmean[j] += classroom[k].position[j]

            for j in range(dim):
                Xmean[j] /= n

            # initialize new solution
            Xnew = [0.0 for i in range(dim)]

            # teaching factor (TF)
            # either 1 or 2 ( randomly chosen)
            TF = random.randint(1, 2)

            # best student of the class is teacher
            Xteacher = Xbest

            # compute new solution
            for j in range(dim):
                Xnew[j] = classroom[i].position[j] + rnd.random() * (Xteacher[j] - TF * Xmean[j])

            # if Xnew < minx OR Xnew > maxx
            # then clip it
            for j in range(dim):
                Xnew[j] = max(Xnew[j], minx)
                Xnew[j] = min(Xnew[j], maxx)

            # compute fitness of new solution
            fnew = fitness(Xnew, func_num)

            # if new solution is better than old
            # replace old with new solution
            if (fnew < classroom[i].fitness):
                classroom[i].position = Xnew
                classroom[i].fitness = fnew

            # update best student
            if (fnew < Fbest):
                Fbest = fnew
                Xbest = Xnew

            ### learning phase of ith student

            # randomly choose a solution from classroom
            # chosen solution should not be ith student
            p = random.randint(0, n - 1)
            while (p == i):
                p = random.randint(0, n - 1)

            # partner solution
            Xpartner = classroom[p]

            Xnew = [0.0 for i in range(dim)]
            if (classroom[i].fitness < Xpartner.fitness):
                for j in range(dim):
                    Xnew[j] = classroom[i].position[j] + rnd.random() * (
                            classroom[i].position[j] - Xpartner.position[j])
            else:
                for j in range(dim):
                    Xnew[j] = classroom[i].position[j] - rnd.random() * (
                            classroom[i].position[j] - Xpartner.position[j])

            # if Xnew < minx OR Xnew > maxx
            # then clip it
            for j in range(dim):
                Xnew[j] = max(Xnew[j], minx)
                Xnew[j] = min(Xnew[j], maxx)

            # compute fitness of new solution
            fnew = fitness(Xnew, func_num)

            # if new solution is better than old
            # replace old with new solution
            if (fnew < classroom[i].fitness):
                classroom[i].position = Xnew
                classroom[i].fitness = fnew

            # update best student
            if (fnew < Fbest):
                Fbest = fnew
                Xbest = Xnew

        Iter += 1
        # -----------King--------------------#
        # Convergence
        conver.append(Fbest)
        # Average fitness
        sum=0
        for i in range(len(classroom)):
            sum=sum+classroom[i].fitness
        ave=sum/len(classroom)
        fit_history.append(ave)
        #search history
        for n in range(len(classroom)):
            history.append(classroom[n].position)
        # -----------King--------------------#


    # end-while

    # return best student from classroom
    return Xbest , conver, history ,fit_history

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
        # print(f"Iter {t+1} Best: {GbestPosition} with Score: {GbestScore}")
    return GbestPosition, GbestScore

# AVOA
def initial1(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j]
    return X


# Calculate fitness values for each Vulture
def CaculateFitness1(X, fun, func_num):
    fitness = fun(X, func_num)
    return fitness


# Sort fitness.
def SortFitness1(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index


# Sort the position of the Vulture according to fitness.
def SortPosition1(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew


# Boundary detection function.
def BorderCheck1(X, lb, ub, dim):
    for j in range(dim):
        if X[j] < lb[j]:
            X[j] = ub[j]
        elif X[j] > ub[j]:
            X[j] = lb[j]
    return X


def rouletteWheelSelection(x):
    CS = np.cumsum(x)
    Random_value = random.random()
    index = np.where(Random_value <= CS)
    index = sum(index)
    return index


def random_select(Pbest_Vulture_1, Pbest_Vulture_2, alpha, betha):
    probabilities = [alpha, betha]
    index = rouletteWheelSelection(probabilities)
    if (index.all() > 0):
        random_vulture_X = Pbest_Vulture_1
    else:
        random_vulture_X = Pbest_Vulture_2
    return random_vulture_X


def exploration(current_vulture_X, random_vulture_X, F, p1, upper_bound, lower_bound):
    if random.random() < p1:
        current_vulture_X = random_vulture_X - (abs((2 * random.random()) * random_vulture_X - current_vulture_X)) * F
    else:
        current_vulture_X = (random_vulture_X - (F) + random.random() * (
                    (upper_bound - lower_bound) * random.random() + lower_bound))
    return current_vulture_X


def exploitation(current_vulture_X, Best_vulture1_X, Best_vulture2_X, random_vulture_X, F, p2, p3, variables_no,
                 upper_bound, lower_bound):
    if abs(F) < 0.5:

        if random.random() < p2:
            A = Best_vulture1_X - ((np.multiply(Best_vulture1_X, current_vulture_X)) / (
                        Best_vulture1_X - current_vulture_X ** 2)) * F
            B = Best_vulture2_X - (
                        (Best_vulture2_X * current_vulture_X) / (Best_vulture2_X - current_vulture_X ** 2)) * F
            current_vulture_X = (A + B) / 2
        else:
            current_vulture_X = random_vulture_X - abs(random_vulture_X - current_vulture_X) * F * levyFlight(
                variables_no)

    if random.random() >= 0.5:
        if random.random() < p3:
            current_vulture_X = (abs((2 * random.random()) * random_vulture_X - current_vulture_X)) * (
                        F + random.random()) - (random_vulture_X - current_vulture_X)

        else:
            s1 = random_vulture_X * (random.random() * current_vulture_X / (2 * np.pi)) * np.cos(current_vulture_X)
            s2 = random_vulture_X * (random.random() * current_vulture_X / (2 * np.pi)) * np.sin(current_vulture_X)
            current_vulture_X = random_vulture_X - (s1 + s2)
    return current_vulture_X

# eq (18)
def levyFlight(d):
    beta = 3 / 2
    sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (
                math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(1, d) * sigma
    v = np.random.randn(1, d)
    step = u / abs(v) ** (1 / beta)
    o = step
    return o

def AVA(pop, dim, lb, ub, Max_iter, fun, func_num):
    alpha = 0.8
    betha = 0.2
    p1 = 0.6
    p2 = 0.4
    p3 = 0.6
    Gama = 2.5
    X = initial1(pop, dim, lb, ub)  # Initialize the random population
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = CaculateFitness1(X[i, :], fun, func_num)
    fitness, sortIndex = SortFitness1(fitness)  # Sort the fitness values of African Vultures

    X = SortPosition1(X, sortIndex)  # Sort the African Vultures population based on fitness
    GbestScore = fitness[0]  # Stores the optimal value for the current iteration.
    GbestPositon = X[0, :][0]
    Xnew = np.zeros([pop, dim])
    # Main iteration starts here
    for t in range(Max_iter):
        Pbest_Vulture_1 = X[0, :]  # location of Vulture (First best location Best Vulture Category 1)
        Pbest_Vulture_2 = X[1, :]  # location of Vulture (Second best location Best Vulture Category 1)
        t3 = np.random.uniform(-2, 2, 1) * (
                    (np.sin((np.pi / 2) * (t / Max_iter)) ** Gama) + np.cos((np.pi / 2) * (t / Max_iter)) - 1)
        z = random.randint(-1, 0)
        # F= (2*random.random()+1)*z*(1-(t/Max_iter))+t3
        P1 = (2 * random.random() + 1) * (1 - (t / Max_iter)) + t3
        F = P1 * (2 * random.random() - 1)
        # For each vulture Pi
        for i in range(pop):
            current_vulture_X = X[i, :]
            random_vulture_X = random_select(Pbest_Vulture_1, Pbest_Vulture_2, alpha,
                                             betha)  # select random vulture using eq(1)
            if abs(F) >= 1:
                current_vulture_X = exploration(current_vulture_X, random_vulture_X, F, p1, ub, lb)  # eq (16) & (17)

            else:
                current_vulture_X = exploitation(current_vulture_X, Pbest_Vulture_1, Pbest_Vulture_2, random_vulture_X,
                                                 F, p2, p3, dim, ub, lb)  # eq (10) & (13)

            Xnew[i, :] = current_vulture_X[0]
            Xnew[i, :] = BorderCheck1(Xnew[i, :], lb, ub, dim)
            tempFitness = CaculateFitness1(Xnew[i, :], fun, func_num)
            # Update local best solution
            if (tempFitness <= fitness[i]):
                fitness[i] = tempFitness
                X[i, :] = Xnew[i, :]
        Ybest, index = SortFitness1(fitness)
        X = SortPosition1(X, index)
        # print('fitness',fitness)
        # -----------King--------------------#
        # history.append(X)
        # for i in range(len(fitness)):
        #     ave = np.average(fitness)
        # fit_history.append(ave)
        # print('ave=',ave)
        # -----------King--------------------#

        # Update global best solution
        if (Ybest[0] <= GbestScore):
            GbestScore = Ybest[0]
            GbestPositon = X[index[0], :][0]
        # print(GbestPositon)
    return GbestPositon, GbestScore


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

#AVOA 30
replication = 1
result_list =[]
runtime_list_out = []
for func_num in range(1):
    Gbest_of_all = []
    runtime_list = []
    pop = 30  # Population size n 50
    MaxIter = 500  # Maximum number of iterations. 100
    dim = 30  # The dimension. 30
    lower, upper = get_lower_upper_bound(func_num)  # The lower and upper bound of the search interval.
    lb = lower * np.ones([dim, 1])
    ub = upper * np.ones([dim, 1])
    for i in range(replication):
        rng = np.random.default_rng()
        time_start = time.time()
        GbestPositon, GbestScore = AVA(pop, dim, lb, ub, MaxIter, fun, func_num)  # Afican Vulture Optimization Algorithm
        time_end = time.time()
        result = fun(GbestPositon, func_num)
        Gbest_of_all.append(result)
        runtime = time_end - time_start
        print(f"【{i} round】\nThe running time is: {runtime} s")
        runtime_list.append(runtime)
        print('The optimal value：', result, "\n=================================")
    print(f'Benchmark: F {func_num+1}\n')
    print(f"Solution Final Output: \n\tnumber of replications={replication}\n\t"
          f"Best of Gbests={np.min(Gbest_of_all)}\n\tWorst of Gbests={np.max(Gbest_of_all)}\n\taverage of Gbests={np.average(Gbest_of_all)}\n\tSTD of Gbests={np.std(Gbest_of_all)}")
    print(f"Runtime Final Output: \n\tnumber of replications={replication}\n\taverage of Runtime={np.average(runtime_list)}\n\t"
          f"Best of Runtime={np.min(runtime_list)}\n\tWorst of Runtime={np.max(runtime_list)}\n\tSTD of Runtime={np.std(runtime_list)}")
    temp_list_result = []
    temp_list_runtime = []
    str_func = 'F'+ str(func_num+1)
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

Gbest_of_each = []
Gbest_of_each.append(Gbest_of_all)
# print(f'result_list: {result_list}')
# print(f'Gbest_of_each: {Gbest_of_each}')
col_list = []
for i in range(len(Gbest_of_all)):
    col_list.append(str(i+1))
df_replication = pd.DataFrame(np.array(Gbest_of_each, dtype='object'), columns= col_list)
# print(df_replication)
# print(np.array(result_list, dtype='object'))
df_result = pd.DataFrame(np.array(result_list, dtype='object'), columns=['Benchmark', 'Mean','Mean_no_outlier', 'Median', 'Best', 'Worst', 'STD'])
df_runtime = pd.DataFrame(np.array(runtime_list_out, dtype='object'), columns=['Benchmark', 'Mean', 'Median', 'Best', 'Worst', 'STD'])

os.makedirs('Data', exist_ok=True)
df_result.to_csv('Data/avoa_result_ou tput_30.csv')
df_runtime.to_csv('Data/avoa_runtime_output_30.csv')
df_replication.to_csv('Data/avoa_replication_output_30.csv')

#AVOA 100
replication = 1
result_list =[]
runtime_list_out = []
for func_num in range(1):
    Gbest_of_all = []
    runtime_list = []
    pop = 30  # Population size n 50
    MaxIter = 500  # Maximum number of iterations. 100
    dim = 100  # The dimension. 30
    lower, upper = get_lower_upper_bound(func_num)  # The lower and upper bound of the search interval.
    lb = lower * np.ones([dim, 1])
    ub = upper * np.ones([dim, 1])
    for i in range(replication):
        rng = np.random.default_rng()
        time_start = time.time()
        GbestPositon, GbestScore = AVA(pop, dim, lb, ub, MaxIter, fun, func_num)  # Afican Vulture Optimization Algorithm
        time_end = time.time()
        result = fun(GbestPositon, func_num)
        Gbest_of_all.append(result)
        runtime = time_end - time_start
        print(f"【{i} round】\nThe running time is: {runtime} s")
        runtime_list.append(runtime)
        print('The optimal value：', result, "\n=================================")
    print(f'Benchmark: F {func_num+1}\n')
    print(f"Solution Final Output: \n\tnumber of replications={replication}\n\t"
          f"Best of Gbests={np.min(Gbest_of_all)}\n\tWorst of Gbests={np.max(Gbest_of_all)}\n\taverage of Gbests={np.average(Gbest_of_all)}\n\tSTD of Gbests={np.std(Gbest_of_all)}")
    print(f"Runtime Final Output: \n\tnumber of replications={replication}\n\taverage of Runtime={np.average(runtime_list)}\n\t"
          f"Best of Runtime={np.min(runtime_list)}\n\tWorst of Runtime={np.max(runtime_list)}\n\tSTD of Runtime={np.std(runtime_list)}")
    temp_list_result = []
    temp_list_runtime = []
    str_func = 'F'+ str(func_num+1)
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

Gbest_of_each = []
Gbest_of_each.append(Gbest_of_all)
# print(f'result_list: {result_list}')
# print(f'Gbest_of_each: {Gbest_of_each}')
col_list = []
for i in range(len(Gbest_of_all)):
    col_list.append(str(i+1))
df_replication = pd.DataFrame(np.array(Gbest_of_each, dtype='object'), columns= col_list)
# print(df_replication)
# print(np.array(result_list, dtype='object'))
df_result = pd.DataFrame(np.array(result_list, dtype='object'), columns=['Benchmark', 'Mean','Mean_no_outlier', 'Median', 'Best', 'Worst', 'STD'])
df_runtime = pd.DataFrame(np.array(runtime_list_out, dtype='object'), columns=['Benchmark', 'Mean', 'Median', 'Best', 'Worst', 'STD'])

os.makedirs('Data', exist_ok=True)
df_result.to_csv('Data/avoa_result_output_100.csv')
df_runtime.to_csv('Data/avoa_runtime_output_100.csv')
df_replication.to_csv('Data/avoa_replication_output_100.csv')

#PSO 30
replication = 1
result_list =[]
runtime_list_out = []

for func_num in range(1):
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
        GbestPositon, GbestScore = PSO(pop, dim, lb, ub, MaxIter, fun, func_num)
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

Gbest_of_each = []
Gbest_of_each.append(Gbest_of_all)
# print(f'result_list: {result_list}')
# print(f'Gbest_of_each: {Gbest_of_each}')
col_list = []
for i in range(len(Gbest_of_all)):
    col_list.append(str(i+1))
df_replication = pd.DataFrame(np.array(Gbest_of_each, dtype='object'), columns= col_list)
df_result = pd.DataFrame(np.array(result_list, dtype='object'), columns=['Benchmark', 'Mean','Mean_no_outlier', 'Median', 'Best', 'Worst', 'STD'])
df_runtime = pd.DataFrame(np.array(runtime_list_out, dtype='object'), columns=['Benchmark', 'Mean', 'Median', 'Best', 'Worst', 'STD'])

os.makedirs('Data', exist_ok=True)
df_result.to_csv('Data/pso_result_output_30.csv')
df_runtime.to_csv('Data/pso_runtime_output_30.csv')
df_replication.to_csv('Data/pso_replication_output_30.csv')

#PSO 100
replication = 1
result_list =[]
runtime_list_out = []

for func_num in range(1):
    print(f"Benchmark: {func_num+1}")
    Gbest_of_all = []
    runtime_list = []
    pop = 500  # Population size n 1000
    MaxIter = 300  # Maximum number of iterations. 500
    dim = 100  # The dimension. 30
    lower, upper = get_lower_upper_bound(func_num)  # The lower and upper bound of the search interval.
    for i in range(replication):
        time_start = time.time()
        history = []
        fit_history = []
        lb = lower * np.ones([dim, 1])
        ub = upper * np.ones([dim, 1])
        GbestPositon, GbestScore = PSO(pop, dim, lb, ub, MaxIter, fun, func_num)
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

Gbest_of_each = []
Gbest_of_each.append(Gbest_of_all)
# print(f'result_list: {result_list}')
# print(f'Gbest_of_each: {Gbest_of_each}')
col_list = []
for i in range(len(Gbest_of_all)):
    col_list.append(str(i+1))
df_replication = pd.DataFrame(np.array(Gbest_of_each, dtype='object'), columns= col_list)
df_result = pd.DataFrame(np.array(result_list, dtype='object'), columns=['Benchmark', 'Mean','Mean_no_outlier', 'Median', 'Best', 'Worst', 'STD'])
df_runtime = pd.DataFrame(np.array(runtime_list_out, dtype='object'), columns=['Benchmark', 'Mean', 'Median', 'Best', 'Worst', 'STD'])

os.makedirs('Data', exist_ok=True)
df_result.to_csv('Data/pso_result_output_100.csv')
df_runtime.to_csv('Data/pso_runtime_output_100.csv')
df_replication.to_csv('Data/pso_replication_output_100.csv')

#TLBO 30
replication = 1
result_list =[]
runtime_list_out = []

for func_num in range(1):
    Gbest_of_all = []
    runtime_list = []
    pop = 100 #100
    MaxIter = 300 #500
    dim = 30 #30
    lower, upper = get_lower_upper_bound(func_num)  # The lower and upper bound of the search interval.
    for i in range(replication):
        conver=[]
        history=[]
        fit_history=[]
        time_start = time.time()
        GbestPositon, conver, history, fit_history = tlbo(fun, func_num, MaxIter, pop, dim, lower, upper, conver,history,fit_history)
        time_end = time.time()
        GbestScore = fun(GbestPositon, func_num)
        Gbest_of_all.append(GbestScore)
        runtime = time_end - time_start
        print(f"【{i} round】\nThe running time is: {runtime} s")
        runtime_list.append(runtime)
        print('The optimal value：', GbestScore, "\n=================================")
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

Gbest_of_each = []
Gbest_of_each.append(Gbest_of_all)
# print(f'result_list: {result_list}')
# print(f'Gbest_of_each: {Gbest_of_each}')
col_list = []
for i in range(len(Gbest_of_all)):
    col_list.append(str(i+1))
df_replication = pd.DataFrame(np.array(Gbest_of_each, dtype='object'), columns= col_list)
df_result = pd.DataFrame(np.array(result_list, dtype='object'), columns=['Benchmark', 'Mean','Mean_no_outlier', 'Median', 'Best', 'Worst', 'STD'])
df_runtime = pd.DataFrame(np.array(runtime_list_out, dtype='object'), columns=['Benchmark', 'Mean', 'Median', 'Best', 'Worst', 'STD'])

os.makedirs('Data', exist_ok=True)
df_result.to_csv('Data/tlbo_result_output_30.csv')
df_runtime.to_csv('Data/tlbo_runtime_output_30.csv')
df_replication.to_csv('Data/tlbo_replication_output_30.csv')

#TLBO 100
replication = 1
result_list =[]
runtime_list_out = []

for func_num in range(1):
    Gbest_of_all = []
    runtime_list = []
    pop = 100 #100
    MaxIter = 300 #500
    dim = 100 #30
    lower, upper = get_lower_upper_bound(func_num)  # The lower and upper bound of the search interval.
    for i in range(replication):
        conver=[]
        history=[]
        fit_history=[]
        time_start = time.time()
        GbestPositon, conver, history, fit_history = tlbo(fun, func_num, MaxIter, pop, dim, lower, upper, conver,history,fit_history)
        time_end = time.time()
        GbestScore = fun(GbestPositon, func_num)
        Gbest_of_all.append(GbestScore)
        runtime = time_end - time_start
        print(f"【{i} round】\nThe running time is: {runtime} s")
        runtime_list.append(runtime)
        print('The optimal value：', GbestScore, "\n=================================")
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

Gbest_of_each = []
Gbest_of_each.append(Gbest_of_all)
# print(f'result_list: {result_list}')
# print(f'Gbest_of_each: {Gbest_of_each}')
col_list = []
for i in range(len(Gbest_of_all)):
    col_list.append(str(i+1))
df_replication = pd.DataFrame(np.array(Gbest_of_each, dtype='object'), columns= col_list)
df_result = pd.DataFrame(np.array(result_list, dtype='object'), columns=['Benchmark', 'Mean','Mean_no_outlier', 'Median', 'Best', 'Worst', 'STD'])
df_runtime = pd.DataFrame(np.array(runtime_list_out, dtype='object'), columns=['Benchmark', 'Mean', 'Median', 'Best', 'Worst', 'STD'])

os.makedirs('Data', exist_ok=True)
df_result.to_csv('Data/tlbo_result_output_100.csv')
df_runtime.to_csv('Data/tlbo_runtime_output_100.csv')
df_replication.to_csv('Data/tlbo_replication_output_100.csv')