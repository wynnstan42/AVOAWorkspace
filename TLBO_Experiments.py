import random
import sys
import time
import numpy as np
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
def F12(X):  # TODO 怪怪的
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
def F13(X):  # TODO 這個也怪怪的
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


# Student class
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


# end teaching learning based optimization

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
            plt.scatter(history[i][n], history[i][n], c="black", alpha=0.3, facecolor='white')
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
    plt.show()


# Execute
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
    Gbest_of_all = []
    runtime_list = []
    pop = 100 #100
    MaxIter = 500 #500
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
    temp_list_result.append(np.average(Gbest_of_all))
    temp_list_result.append(np.min(Gbest_of_all))
    temp_list_result.append(np.max(Gbest_of_all))
    temp_list_result.append(np.std(Gbest_of_all))

    temp_list_runtime.append(str_func)
    temp_list_runtime.append(np.average(runtime_list))
    temp_list_runtime.append(np.min(runtime_list))
    temp_list_runtime.append(np.max(runtime_list))
    temp_list_runtime.append(np.std(runtime_list))

    result_list.append(temp_list_result)
    runtime_list_out.append(temp_list_runtime)

    print(Gbest_of_all)

print(np.array(result_list, dtype='object'))
df_result = pd.DataFrame(np.array(result_list, dtype='object'), columns=['Benchmark', 'Mean', 'Best', 'Worst', 'STD'])
df_runtime = pd.DataFrame(np.array(runtime_list_out, dtype='object'), columns=['Benchmark', 'Mean', 'Best', 'Worst', 'STD'])

import os
os.makedirs('Data', exist_ok=True)
df_result.to_csv('Data/tlbo_result_output.csv')
df_runtime.to_csv('Data/tlbo_runtime_output.csv')

# plot_convergrnce(conver)
# plot_search_history(history)
# plot_fitness(fit_history)
