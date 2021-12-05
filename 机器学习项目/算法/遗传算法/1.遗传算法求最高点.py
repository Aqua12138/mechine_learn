import random
import numpy as np
import matplotlib.pyplot as plt
DNA_SIZE = 10
POP_SIZE = 100
CROSS_RATE = 0.8
MUTATION = 0.03
N_GENERATION = 200
X_BOUND = [0,5]
#环境
def F(x):
    return np.sin(10*x)*x + np.cos(2*x)*x
#后代遗传特征
def get_fitness(pred):
    return np.clip(pred,0.1,10)
#父代基因交叉配对
def crossover(parent,pop):
    if np.random.rand() < CROSS_RATE:
        i = np.random.randint(0,POP_SIZE,size = 1)
        cross_points = np.random.randint(0,2,size = DNA_SIZE).astype(np.bool_)
        parent[cross_points] = pop[i,cross_points]#两边同时取bool运算，相等的相等，不相等的为0
    return parent
#变异
def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand()<MUTATION:
            if child[point] == 1:
                child[point] = 0
            else:
                child[point] = 1
    return child
#适者生存，淘汰差后代
def select(pop,fitness):
    idx = np.random.choice(np.arange(POP_SIZE),size=POP_SIZE,replace=True,p=fitness/fitness.sum())
    return pop[idx]
def translationDNA(pop):
    return pop.dot(2 ** np.arange(DNA_SIZE)[::-1])/(2 ** DNA_SIZE -1) * X_BOUND[1]

pop = np.random.randint(0,2,(1,10)).repeat(POP_SIZE,axis=0)
plt.ion()
x = np.linspace(*X_BOUND, 200)
for i in range(N_GENERATION):
    F_values = F(translationDNA(pop))
    plt.cla()
    plt.scatter(translationDNA(pop), F_values, s=200, lw=0, c='red', alpha=0.5)
    plt.plot(x, F(x))
    plt.pause(0.05)
    fitness = get_fitness(F_values)
    pop = select(pop,fitness)#选择特征比较
    pop_copy = pop.copy()
    for parent in pop:
        child = crossover(parent,pop_copy)
        child = mutate(child)
        parent[:] = child#覆盖

plt.ioff()
plt.show()
