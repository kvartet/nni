import random
import heapq
import numpy as np
import matplotlib.pyplot as plt
import copy
import itertools


from sko.GA import GA
from sklearn import metrics
from scipy.stats import variation 
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

class cartesian(object):
    def __init__(self):
        self._data_list=[]

    def add_data(self,data=[]): #添加生成笛卡尔积的数据列表
        self._data_list.append(data)

    def build(self): #计算笛卡尔积
        item_list = []
        for item in itertools.product(*self._data_list):
            item_list.append(item)
        return item_list

def conv(in_ch, out_ch, input_size, kernel_size=1, stride=1, padding=0, groups=1, bias=False):
    output_size = (input_size-kernel_size+2*padding)//stride + 1
    flops = (pow(kernel_size, 2) * pow(output_size, 2)
             * in_ch * out_ch) // groups
    if bias:
        flops += pow(output_size, 2)*out_ch
    return flops


def pool(in_ch, input_size):
    flops = pow(input_size, 2) * in_ch
    return flops


def bn(in_ch, input_size, d=True):
    flops = pow(input_size, 2) * in_ch
    if d:
        flops *= 2
    return flops


def relu(in_ch, input_size):
    flops = pow(input_size, 2) * in_ch
    return flops


def se(in_ch, input_size):
    # (mid_ch, output_size)
    divisor = 8
    mid_ch = int(in_ch//4 + divisor / 2) // divisor * divisor
    flops = 0
    flops += pool(in_ch, input_size)
    flops += conv(in_ch, mid_ch, 1, bias=True)
    flops += relu(mid_ch, 1)
    flops += conv(mid_ch, in_ch, 1, bias=True)
    return flops


def ds():
    flops = 0
    flops += conv(16, 16, 3, 1, 1, 16, 112)
    flops += bn(16, 112)
    # flops += act(16,112)
    flops += se(16, 112)
    flops += conv(16, 16, 1, 1, 0, 1, 112)
    flops += bn(16, 112)
    print('flops {}'.format(flops))


def ir(in_ch, out_ch, kernel_size, exp_rate, stride, input_size, padding=1):
    flops = 0
    padding = (kernel_size-1)//2
    mid_ch = in_ch*exp_rate
    flops += conv(in_ch, mid_ch, input_size=input_size)
    flops += bn(mid_ch, input_size)
    # flops += act(64,112) # Swish()
    flops += conv(mid_ch, mid_ch, input_size, kernel_size=kernel_size,
                  stride=stride, padding=padding, groups=mid_ch)
    output_size = (input_size-kernel_size+2*padding)//stride + 1
    flops += bn(mid_ch, output_size)
    flops += se(mid_ch, output_size)
    # flops += act(64,56) # Swish
    flops += conv(mid_ch, out_ch, input_size=output_size)
    flops += bn(out_ch, output_size)
    # print('ir flops {}'.format(flops))
    return flops


def coefficient(in_ch, out_ch, input_size, stride, r=0.25, k=3, e=4):
    a = in_ch*pow(input_size, 2)/pow(stride, 2)
    b = 2*pow(in_ch, 2)*r
    c = pow(in_ch, 2)*pow(input_size, 2)+2*in_ch*pow(input_size, 2) + \
        3*in_ch*pow(input_size, 2)/pow(stride, 2)+2*r*in_ch + \
        in_ch+in_ch*out_ch*pow(input_size, 2)/pow(stride, 2)
    d = 2*out_ch*pow(input_size, 2)/pow(stride, 2)
    flops = a*pow(k, 2)*e+b*pow(e, 2)+c*e+d
    print('a {}, b {}, c {}, d {}'.format(a/1e6, b/1e6, c/1e6, d/1e6))
    # print('flops {}'.format(flops))
    return [a/1e6, b/1e6, c/1e6, d/1e6]
    # 21828704

def cal_coefficient():
    coefficient_list = []
    channels_list = [16, 24, 40, 80, 96, 192]
    strides_list = [2, 2, 2, 1, 2]
    block_list = [4, 4, 5, 4, 4]
    input_size = 112
    stage_num = 5
    index = 0

    for stage in range(stage_num):
        # block 0
        coefficient_list.append(coefficient(in_ch=channels_list[stage], out_ch=channels_list[stage+1],input_size=input_size,stride=strides_list[stage]))
        input_size = input_size // strides_list[stage]
        index += 1
        # block 1,2,3,4
        for i in range(1, block_list[stage]):
            coefficient_list.append(coefficient(in_ch=channels_list[stage+1], out_ch=channels_list[stage+1],input_size=input_size,stride=1))

    return coefficient_list



def flops_table():
    # initialize
    choices = {'kernel_size': [3, 5, 7], 'exp_ratio': [4, 6]}
    choices_list = [[x, y] for x in choices['kernel_size']
                    for y in choices['exp_ratio']]
    channels_list = [16, 24, 40, 80, 96, 192]
    strides_list = [2, 2, 2, 1, 2]
    block_list = [4, 4, 5, 4, 4]
    input_size = 112
    stage_num = 5
    index = 0
    flops_dict = [[] for _ in range(21)]

    for stage in range(stage_num):
        # block 0
        for choice in choices_list:
            flops = ir(in_ch=channels_list[stage], out_ch=channels_list[stage+1],
                       kernel_size=choice[0], exp_rate=choice[1], stride=strides_list[stage], input_size=input_size)
            # print('input_size {}, input_channels {}, stage {}, block 0, choice {}, flops {}'.format(
            # input_size, channels_list[stage], stage+1, choice, flops))
            flops_dict[index].append(flops)
        input_size = input_size // strides_list[stage]
        index += 1
        # block 1,2,3,4
        for i in range(1, block_list[stage]):
            for choice in choices_list:
                flops = ir(in_ch=channels_list[stage+1], out_ch=channels_list[stage+1],
                           kernel_size=choice[0], exp_rate=choice[1], stride=1, input_size=input_size)
                # print('input_size {}, input_channels {}, stage {}, block {}, choice {}, flops {}'.format(
                # input_size, channels_list[stage+1], stage+1, i, choice, flops))
                flops_dict[index].append(flops)
            index += 1
        # print()
    return flops_dict


def pred(p,upper_bound):
    # initialization
    total = 0
    block_num = 21
    
    choices_dict = {} # 
    choices_list = [[x, y] for x in [3,5,7] for y in [4,6]]
    for i in range(6):
        choices_dict[i+1] = choices_list[i]
    sample = []
    for i in range(block_num):
        sample.extend(choices_dict[p[i]])
    for p in range(len(upper_bound)): # 表示空间里共有n个并集
        flag = 1
        for i in range(block_num*2):
            if sample[i] > int(upper_bound[p][i]):
                flag = 0
        if flag == 1:
            total = 1
            break
    # print('sample {}'.format(sample))
    # print([int(i) for i in upper_bound])
    return total
        
def satisfaction(flops_dict,max_flops,upper_bound):
    '''
    sample 10000次，把实际落在这个区间的当作ground_truth
    然后判断我自己预测的准确不准确
    计算精确率和recall
    没有计算 flops_fixed
    '''
    # min 308.8329, max 591.7952
    # flops_fixed = 14908896.0/1e6
    flops_stats = []
    y_true, y_pred = [], []
    sample_num = 10000
    block_num = 21
    for _ in range(sample_num):
        rand = [random.randint(1,6) for _ in range(block_num)]
        flops = 0
        for i in range(block_num):
            flops += flops_dict[i][rand[i]-1]
        y_true.append(1 if flops/1e6 < max_flops else 0)
        y_pred.append(pred(rand,upper_bound))
    # print(y_true,y_pred)
    precision = metrics.precision_score(y_true,y_pred)
    recall = metrics.recall_score(y_true,y_pred)
    f1 = metrics.f1_score(y_true,y_pred)
    cm = metrics.confusion_matrix(y_true, y_pred)
    print('precision {:.4f}, recall {:.4f}, f1 {:.4f}'.format(precision,recall,f1))
    print('{}'.format(cm))
    return f1

def visual(flops_dict):
    max_flops = 400
    sample_num = 2000
    block_num = 21
    size = (sample_num,block_num)
    X = np.random.randint(0,6,size)
    y_true = []
    for p in range(sample_num):
        flops = 0
        rand = X[p]
        for i in range(block_num):
            flops += flops_dict[i][rand[i]]
        y_true.append(1 if flops/1e6 < max_flops else 0)
    y_true = np.array(y_true)
    pca = PCA(n_components=3)
    pca = pca.fit(X)
    X_dr = pca.transform(X) 
    colors = ['red', 'blue'] 
    label = ['false','true']
    # plt.figure()
    ax = plt.subplot(projection='3d')

    for i in [0, 1]:
        ax.scatter(X_dr[np.where(y_true == i), 0], X_dr[np.where(y_true == i), 1],X_dr[np.where(y_true == i), 2]
                    ,alpha=.7,c=colors[i] ,label=label[i],s=1) 
    plt.legend()
    plt.title('sample PCA') 
    plt.savefig('PCA2.png')
    plt.show()


def gene_rand(sample_num):
    size = (21,sample_num)
    choice1, choice2 = [3,5,7], [4,6]
    random1 = np.random.randint(0,len(choice1),size)
    random2 = np.random.randint(0,len(choice2),size)
    for i in range(len(choice1)):
        random1[random1==i] = choice1[i]
    for i in range(len(choice2)):
        random2[random2==i] = choice2[i]
    return random1,random2

def yingshe(a,b):
    # [3,4],[3,6],[5,4],[5,6],[7,4],[7,6]
    if a == 5 and b == 4:
        return 2
    elif a == 5 and b == 6:
        return 3
    elif a == 7 and b == 4:
        return 4
    elif a == 7 and b == 6:
        return 5

def cal_var(flops_dict,max_flops,rate):
    sample_num = 10000
    block_num = 21
    satisfy = []
    upper_bound = [7,6]*block_num
    purn_num = block_num
    choices_dict = {} # recall 的意思是
    choices_list = [[x, y] for x in [3,5,7] for y in [4,6]]
    for i in range(6):
        choices_dict[i+1] = choices_list[i]
    for _ in range(sample_num):
        rand = [random.randint(1,6) for _ in range(block_num)]
        rand_tmp = []
        flops = 0
        for i in range(block_num):
            flops += flops_dict[i][rand[i]-1]
        if flops/1e6 < max_flops:
            for i in range(block_num):
                rand_tmp.extend(choices_dict[rand[i]])
            satisfy.append(rand_tmp)
            # print(rand_tmp)
            # print(rand)
    satisfy_num = len(satisfy)
    satisfy=np.array(satisfy)
    var = variation(satisfy, axis=0)
    # small_index = np.argpartition(var, purn_num)[:purn_num]
    # small_index = small_index[np.argsort(var[small_index])]
    small_index = np.argsort(var)[:purn_num]
    for p in range(purn_num):
        upper_bound[small_index[p]] -= 2
        flops = 0
        for i in range(block_num):
            # print(yingshe(upper_bound[2*i],upper_bound[2*i+1]))
            flops += flops_dict[i][yingshe(upper_bound[2*i],upper_bound[2*i+1])]
        # print(flops/1e6)
        if flops/1e6 < max_flops*rate:
            purn_num = p
            break
    # print(purn_num)
    # print(small_index)
    # print(upper_bound)
    purn_index = small_index[:purn_num]
    dict_map = {0: [0], 1: [1], 2: [2, 4, 6], 3: [3, 5, 7], 4: [2, 4, 6], 5: [3, 5, 7], 6: [2, 4, 6], 7: [3, 5, 7], 8: [8], 9: [9], 10: [10, 12, 14], 11: [11, 13, 15], 12: [10, 12, 14], 13: [11, 13, 15], 14: [10, 12, 14], 15: [11, 13, 15], 16: [16], 17: [17], 18: [18, 20, 22, 24], 19: [19, 21, 23, 25], 20: [18, 20, 22, 24], 21: [19, 21, 23, 25], 22: [18, 20, 22, 24], 23: [19, 21, 23, 25], 24: [18, 20, 22, 24], 25: [19, 21, 23, 25], 26: [26], 27: [27], 28: [28, 30, 32], 29: [29, 31, 33], 30: [28, 30, 32], 31: [29, 31, 33], 32: [28, 30, 32], 33: [29, 31, 33], 34: [34], 35: [35], 36: [36, 38, 40], 37: [37, 39, 41], 38: [36, 38, 40], 39: [37, 39, 41], 40: [36, 38, 40], 41: [37, 39, 41]}
    stand = [7,6]*block_num
    # 计算所有解的对偶解
    upper_bound_list = []
    value_list = [upper_bound[i] for i in purn_index]
    car=cartesian()
    for item in purn_index:
        car.add_data(dict_map[item])
    product = car.build()
    for item in product:
        tmp = copy.deepcopy(stand)
        for i in range(purn_num):
            tmp[item[i]]=value_list[i]
        idx_num = 0
        for idx in range(block_num*2):
            if tmp[idx] != stand[idx]:
                idx_num += 1
        if idx_num == purn_num:
            upper_bound_list.append(tmp)
    upper_bound_list = [list(t) for t in set(tuple(_) for _ in upper_bound_list)]
    print('=======================================')
    print('purn_index {}'.format(purn_index))
    print('max_flops:{}, rate:{}'.format(max_flops,rate)) 
    print('choice:{}'.format(len(upper_bound_list)))
    for i in upper_bound_list:
        print('upper_bound_list {}'.format(i))
    f1 = satisfaction(flops_dict,max_flops,upper_bound_list)
    return f1,len(upper_bound_list)



if __name__ == '__main__':
    p = [1.25]
    f1 = 0
    choice = 0
    bestp=0
    for _ in range(10):
        for i in p:
            tmp,tmp2 = cal_var(flops_table(),450,i)
            if tmp > f1:
                f1 = tmp
                choice = tmp2
                bestp = i
    print('best f1:{}, choice:{}, p:{}'.format(f1,choice,bestp))

# 450 1.25
