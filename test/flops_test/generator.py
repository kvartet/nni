import json
from itertools import combinations
import argparse

branch = {
    "branch": {
        "_type": "choice", 
        "_value": []}
}

op_list = ["0_1", "0_2", "0_3", "1_2", "1_3", "2_3"]

def flops_30(d):
    # construct the search space of cifar10/cifar100 when flops = 30
    op1 = {"_type": "choice", "_value": ["none", "skip_connect", "avg_pool_3x3", "conv_1x1"]}
    op2 = {"_type": "choice", "_value": ["none", "skip_connect", "avg_pool_3x3"]}    
    for i in range(6):
        tmp = dict()
        tmp[op_list[i]] = op2
        for j in range(6):
            if i != j:
                tmp[op_list[j]] = op1
        tmp = sorted(tmp.items(), key=lambda x:x[0])
        tmp.insert(0,("_name","choice"+str(i+1)))
        d['branch']['_value'].append(dict(tmp))
    return d


def flops_50(d):
    # construct the search space of cifar10/cifar100 when flops = 50 
    op1 = {"_type": "choice", "_value": ["none", "skip_connect", "avg_pool_3x3", "conv_1x1", "conv_3x3"]}
    op2 = {"_type": "choice", "_value": ["none", "skip_connect", "avg_pool_3x3", "conv_1x1"]}
    op3 = {"_type": "choice", "_value": ["none", "skip_connect", "avg_pool_3x3"]}

    g = {
        "_name": "choice0",
        "0_1": {"_type": "choice", "_value": ["none", "skip_connect", "avg_pool_3x3", "conv_1x1"]},
        "0_2": {"_type": "choice", "_value": ["none", "skip_connect", "avg_pool_3x3", "conv_1x1"]},
        "0_3": {"_type": "choice", "_value": ["none", "skip_connect", "avg_pool_3x3", "conv_1x1"]},
        "1_2": {"_type": "choice", "_value": ["none", "skip_connect", "avg_pool_3x3", "conv_1x1"]},
        "1_3": {"_type": "choice", "_value": ["none", "skip_connect", "avg_pool_3x3", "conv_1x1"]},
        "2_3": {"_type": "choice", "_value": ["none", "skip_connect", "avg_pool_3x3", "conv_1x1"]}
    }
    for i in range(6):
        for j in range(6):
            if i != j:
                tmp = {}
                # tmp = {"_name":"choice"+str(i+1)+'_'+str(j+1)}
                tmp[op_list[i]] = op1
                tmp[op_list[j]] = op2
                for k in range(6):
                    if k != i and k != j:
                        tmp[op_list[k]] = op3
                tmp = sorted(tmp.items(), key=lambda x:x[0])
                tmp.insert(0,("_name","choice"+str(i+1)+'_'+str(j+1)))
                d['branch']['_value'].append(dict(tmp))

    d['branch']['_value'].append(g)
    return d

 
def flops_100(d):
    # construct the search space of cifar10/cifar100 when flops = 100
    op1 = {"_type": "choice", "_value": ["none", "skip_connect", "avg_pool_3x3", "conv_1x1", "conv_3x3"]}
    op2 = {"_type": "choice", "_value": ["none", "skip_connect", "avg_pool_3x3", "conv_1x1"]}
    for c in combinations(op_list, 2):
        # print(c)
        tmp = {}
        tmp[c[0]], tmp[c[1]] = op1, op1
        for item in op_list:
            if item not in c:
                tmp[item] = op2
        tmp = sorted(tmp.items(), key=lambda x:x[0])
        tmp.insert(0,("_name","choice"+str(c[0])+'_'+str(c[1])))
        d['branch']['_value'].append(dict(tmp))
    print(len(d['branch']['_value']))
    return d

# imagenet flops range: 1.9534, 7.85164, 55.03756

def flops_5(d):
    # construct the search space of imagenet when flops = 5
    # the search space including (3 none + 3 conv1x1)
    op1 = {"_type": "choice", "_value": ["none", "skip_connect", "avg_pool_3x3"]}
    op2 = {"_type": "choice", "_value": ["none", "skip_connect", "avg_pool_3x3", "conv_1x1"]}
    op3 = {"_type": "choice", "_value": ["none", "skip_connect", "avg_pool_3x3", "conv_1x1", "conv_3x3"]}
    for c in combinations(op_list, 3):
        tmp = {}
        tmp[c[0]], tmp[c[1]], tmp[c[2]] = op1, op1, op1
        for item in op_list:
            if item not in c:
                tmp[item] = op2
        tmp = sorted(tmp.items(), key=lambda x:x[0])
        tmp.insert(0,("_name","choice"+str(c[0])+'_'+str(c[1])+'_'+str(c[2])))
        d['branch']['_value'].append(dict(tmp))
    print(len(d['branch']['_value']))
    return d       

def flops_11(d):
    # construct the search space of imagenet when flops = 11
    # the search space including (1 conv3x3 + 5 none) or (6 conv1x1)
    op1 = {"_type": "choice", "_value": ["none", "skip_connect", "avg_pool_3x3"]}
    op2 = {"_type": "choice", "_value": ["none", "skip_connect", "avg_pool_3x3", "conv_1x1"]}
    op3 = {"_type": "choice", "_value": ["none", "skip_connect", "avg_pool_3x3", "conv_1x1", "conv_3x3"]}
    for c in combinations(op_list, 1):
        tmp = {}
        tmp[c[0]] = op3
        for item in op_list:
            if item not in c:
                tmp[item] = op1
        tmp = sorted(tmp.items(), key=lambda x:x[0])
        tmp.insert(0,("_name","choice"+str(c[0])))
        d['branch']['_value'].append(dict(tmp))
    tmp = {}
    for item in op_list:
        tmp[item] = op2
    tmp = sorted(tmp.items(), key=lambda x:x[0])
    tmp.insert(0,("_name","choice_conv1"))
    d['branch']['_value'].append(dict(tmp))
    print(len(d['branch']['_value'])) 
    return d        

def flops_20(d):
    # construct the search space of imagenet when flops = 20
    # the search space including (2 conv3x3 + 4 none) or (1 conv3x3 + 5 conv1x1) or (6 conv1x1)
    op1 = {"_type": "choice", "_value": ["none", "skip_connect", "avg_pool_3x3"]}
    op2 = {"_type": "choice", "_value": ["none", "skip_connect", "avg_pool_3x3", "conv_1x1"]}
    op3 = {"_type": "choice", "_value": ["none", "skip_connect", "avg_pool_3x3", "conv_1x1", "conv_3x3"]}
    for c in combinations(op_list, 2):
        tmp = {}
        tmp[c[0]], tmp[c[0]] = op3, op3
        for item in op_list:
            if item not in c:
                tmp[item] = op1
        tmp = sorted(tmp.items(), key=lambda x:x[0])
        tmp.insert(0,("_name","choice"+str(c[0])+'_'+str(c[1])))
        d['branch']['_value'].append(dict(tmp))
    for c in combinations(op_list, 1):
        tmp = {}
        tmp[c[0]] = op3
        for item in op_list:
            if item not in c:
                tmp[item] = op2
        tmp = sorted(tmp.items(), key=lambda x:x[0])
        tmp.insert(0,("_name","choice"+str(c[0])))
        d['branch']['_value'].append(dict(tmp))
    tmp = {}
    for item in op_list:
        tmp[item] = op2
    tmp = sorted(tmp.items(), key=lambda x:x[0])
    tmp.insert(0,("_name","choice_conv1"))
    d['branch']['_value'].append(dict(tmp))
    print(len(d['branch']['_value'])) 
    return d 


parser = argparse.ArgumentParser(description='controller')
parser.add_argument('--flops', type=int, default='100')
args = parser.parse_args()


json_str = json.dumps(globals().get('flops_%s' % args.flops)(branch), indent=4)
# json_str = json.dumps(flops_100(branch), indent=4)
with open('search.json', 'w') as json_file:
    json_file.write(json_str)