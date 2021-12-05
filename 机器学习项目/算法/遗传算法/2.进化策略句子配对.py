import numpy as np
from collections import Counter
# 超参数
TARGET = 'You got it'#目标值
DNA_SIZE = len(TARGET)
POP_SIZE = 1 #总群数量
MUT_STRENGTH = 8
N_GENERATIONS = 1000
#适应度函数

def get_fitness(pred):
    pred = list(np.abs(np.array(pred).astype(np.int)-np.array([ord(j) for j in TARGET])))
    return pred
#产生后代
def make_baby(parent):
    kid = parent + MUT_STRENGTH*np.random.randn(POP_SIZE,DNA_SIZE).astype(int)
    kid = np.clip(kid[0],0,122)
    return kid
#不良特征后代减少
def TranslationDNA(kid):
    kid = ''.join([chr(int(i)) for i in kid])
    return kid
def kill_baby(parent,kid):
    global MUT_STRENGTH
    fp = get_fitness(parent)
    fk = get_fitness(kid)
    p_target = 1/5
    for i in range(DNA_SIZE):
        if fp[i] > fk[i]:
            parent[i] = kid[i]
    if Counter(get_fitness(parent))[0] > Counter(fp)[0]:
        ps = 1
    else:
        ps = 0
    MUT_STRENGTH *= np.exp(1 / np.sqrt(DNA_SIZE + 1) * (ps - p_target) / (1 - p_target))
    return parent
parent = [ord(i) for i in 'I love you']
for _ in range(N_GENERATIONS):
    # ES part
    kid = make_baby(parent)
    parent = kill_baby(parent, kid)
    print(TranslationDNA(parent))

