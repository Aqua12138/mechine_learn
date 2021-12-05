import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as py
#超参数
DNA_SIZE = 20   #DNA长度
POP_SIZE = 500  #每一代的总群数量
GENERATION_NUMBER = 500 #繁衍的后代总数
MUTATION = 0.05 #后代变异的概率
CROSSOVER = 0.02 #后代的杂交配对概率,若没有杂交，则保留母代特征

#遗传算法
class GA():
    #变量初始化
    def __init__(self,DNA_SIZE,POP_SIZE,GENERATION_NUMBER,MUTATION,CROSSOVER):
        self.DNA_SIZE = DNA_SIZE
        self.POP_SIZE = POP_SIZE
        self.GENERATION_NUMBER = GENERATION_NUMBER
        self.MUTATION = MUTATION
        self.CROSSOVER = CROSSOVER
        self.pop = np.vstack([np.random.permutation(DNA_SIZE) for i in range(POP_SIZE)])
    # 适应度函数
    def get_fitness(self,line_x,line_y):
        distance = np.empty([self.POP_SIZE],dtype=np.float64)
        sqrt = np.sqrt(np.square(np.diff(line_x))+np.square(np.diff(line_y)))
        for i in range(self.POP_SIZE):
            distance[i] = sum(sqrt[i])
        fitness = np.exp(self.DNA_SIZE * 2 / distance)
        return fitness, distance
    # 杂交函数
    def cross_over(self,parent,pop):
        if np.random.rand() < self.CROSSOVER:
            bool = np.random.randint(0,2,size=self.DNA_SIZE).astype(np.bool_)
            i = np.random.randint(0,self.POP_SIZE,size=1)
            mother_swap = parent[bool]
            father_swap = pop[i,np.isin(pop[i].ravel(),mother_swap,invert=True)]
            parent[:] = np.concatenate((mother_swap,father_swap))
        return parent
    #产生后代
    def translation(self,city_position,POP_DNA):#city_position:城市x,y的位置矩阵  POP_DNA：存储一代DNA的矩阵
        line_x = np.empty_like(POP_DNA,dtype=np.float64)
        line_y = np.empty_like(POP_DNA,dtype=np.float64)
        for i,d in enumerate(POP_DNA):
            person_position = city_position[d]
            line_x[i,:] = person_position[:,0]
            line_y[i,:] = person_position[:,1]
        return line_x, line_y            #输出整个总群DNA对应的x，y坐标
    #选取一总群的优质基因，准备配种
    def select_pop(self,fitness):
        idx = np.random.choice(np.arange(self.POP_SIZE),size =self.POP_SIZE,replace=True,p=fitness/fitness.sum())
        return self.pop[idx]
    #变异
    def mutate(self,kid):
        for i in range(self.DNA_SIZE):
            if np.random.rand()<self.MUTATION:
                A = np.random.randint(0,self.DNA_SIZE)
                swap1,swap2 = kid[i],kid[A]
                kid[i],kid[A] = swap2,swap1
        return kid
    #进化
    def envolution(self,fitness):
        pop = self.select_pop(fitness)#创建准备交配的总群
        pop_copy = pop.copy()#创建母体
        for parent in pop:
            kid = self.cross_over(parent,pop_copy)
            kid = self.mutate(kid)
            parent[:] = kid
        self.pop = pop
#显示环境
class env():
    def __init__(self,CITY_N):
        self.city_position = np.random.rand(CITY_N,2)
        plt.ion()
    def plotting(self, lx, ly, total_d):
        plt.cla()
        plt.scatter(self.city_position[:, 0].T, self.city_position[:, 1].T, s=100, c='k')
        plt.plot(lx.T, ly.T, 'r-')
        plt.text(-0.05, -0.05, "Total distance=%.2f" % total_d, fontdict={'size': 20, 'color': 'red'})
        plt.xlim((-0.1, 1.1))
        plt.ylim((-0.1, 1.1))
        plt.pause(0.01)
#主函数
ga = GA(DNA_SIZE=DNA_SIZE,POP_SIZE=POP_SIZE,GENERATION_NUMBER=GENERATION_NUMBER,MUTATION=MUTATION,CROSSOVER=CROSSOVER)
env = env(CITY_N=DNA_SIZE)
for i in range(GENERATION_NUMBER):
    linex,liney = ga.translation(env.city_position,ga.pop)
    fitness,distance = ga.get_fitness(linex,liney)
    ga.envolution(fitness)
    best_idx = np.argmax(fitness)
    print('Gen:', i, '| best fit: %.2f' % fitness[best_idx], )

    env.plotting(linex[best_idx], liney[best_idx], distance[best_idx])

plt.ioff()
plt.show()



