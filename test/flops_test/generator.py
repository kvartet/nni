import json
from itertools import combinations

branch = {
    "branch": {
        "_type": "choice", 
        "_value": []}
}

op_list = ["0_1", "0_2", "0_3", "1_2", "1_3", "2_3"]

def flops_30(d):
    # construct the search space when flops = 30
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
    # construct the search space when flops = 50 
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
    # construct the search space when flops = 100
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
        
json_str = json.dumps(flops_50(branch), indent=4)
with open('search.json', 'w') as json_file:
    json_file.write(json_str)