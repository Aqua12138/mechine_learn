import numpy as np
# 超参数
TARGET = 'I miss you'#目标值
DNA_SIZE = len(TARGET)
POP_SIZE =100 #总群数量
MUT_STRENGTH = 8
N_GENERATIONS = 1000
TARGET_ASCII = np.fromstring(TARGET,dtype = np.uint8)
ASCII_Bound = [32,126]
CROSS_RATE = 0.8
MUTATION = 0.03
pop = np.random.randint(*ASCII_Bound,(1,10)).repeat(POP_SIZE,axis=0)
#环境
#后代遗传特征
def get_fitness(pop):
    match = (pop == TARGET_ASCII).sum(axis=1)+0.001
    return match
def crossover(parent,pop):
    if np.random.rand() < CROSS_RATE:
        i = np.random.randint(0,POP_SIZE,size = 1)
        cross_points = np.random.randint(0,2,size = DNA_SIZE).astype(np.bool)
        parent[cross_points] = pop[i,cross_points]#两边同时取bool运算，相等的相等，不相等的为0
    return parent
#变异
def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand()<MUTATION:
            child[point] = np.random.randint(*ASCII_Bound)
    return child
#适者生存，淘汰差后代
def select(pop,fitness):
    idx = np.random.choice(np.arange(POP_SIZE),size=POP_SIZE,replace=True,p=fitness/(fitness.sum()))
    return pop[idx]
def translationDNA(pop):
    return ''.join([chr(i) for i in pop])
def evolve(pop,fitness):
    pop = select(pop,fitness)
    pop_copy = pop.copy()
    for parent in pop:
        child = crossover(parent,pop_copy)
        child = mutate(child)
        parent[:] = child
    return pop

for i in range(N_GENERATIONS):
    best_DNA = pop[np.argmax(get_fitness(pop))]
    best_prase = translationDNA(best_DNA)
    print('GEN',i,':',best_prase)
    if best_prase == TARGET:
        break
    pop = evolve(pop,get_fitness(pop))


