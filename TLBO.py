import random
import sys
import time

import numpy as np
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


# Student class
class Student:
    def __init__(self, fitness, dim, minx, maxx, seed):
        self.rnd = random.Random(seed)

        # a list of size dim
        # with 0.0 as value of all the elements
        self.position = [0.0 for i in range(dim)]

        # loop dim times and randomly select value of decision var
        # value should be in between minx and maxx
        for i in range(dim):
            self.position[i] = ((maxx - minx) *
                                self.rnd.random() + minx)

        # compute the fitness of student
        self.fitness = fitness(self.position)


# Teaching learning based optimization
def tlbo(fitness, max_iter, n, dim, minx, maxx, conver, history, fit_history):
    rnd = random.Random(0)

    # create n random students
    classroom = [Student(fitness, dim, minx, maxx, i) for i in range(n)]

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
        if Iter % 10 == 0 and Iter > 1:
            print("Iter = " + str(Iter) + " best fitness = %f" % Fbest)

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
            fnew = fitness(Xnew)

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
            fnew = fitness(Xnew)

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
        sum = 0
        for i in range(len(classroom)):
            sum = sum + classroom[i].fitness
        ave = sum / len(classroom)
        fit_history.append(ave)
        # search history
        for n in range(len(classroom)):
            history.append(classroom[n].position)
        # -----------King--------------------#

    # end-while

    # return best student from classroom
    return Xbest, conver, history, fit_history


# end teaching learning based optimization

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
def fun(X):
    return F13(X)


time_start = time.time()
pop = 100  # 100
MaxIter = 500  # 500
dim = 30  # 30
lower = -50
upper = 50
conver = []
history = []
fit_history = []
GbestPositon, conver, history, fit_history = tlbo(fun, MaxIter, pop, dim, lower, upper, conver, history, fit_history)
GbestScore = fun(GbestPositon)
time_end = time.time()
# print('len=',len())
print(f"The running time is: {time_end - time_start} s")
print('The optimal value：', GbestScore)
print('The optimal solution：', GbestPositon)
plot_convergrnce(conver)
plot_search_history(history)
plot_fitness(fit_history)
