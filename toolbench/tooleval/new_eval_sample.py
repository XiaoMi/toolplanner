# Copyright (C) 2024 Xiaomi Corporation.
# The source code included in this project is licensed under the Apache 2.0 license.

import json
import os
import re
import sys

eval_dir = sys.argv[1]
while (eval_dir.endswith('/')):
    eval_dir = eval_dir[ : -1]

eval_id = eval_dir.split('_')[-1]

'''
if 'G1' in eval_dir:
    eval_type = 'G1'
elif 'G2' in eval_dir:
    eval_type = 'G2'
elif 'G3' in eval_dir:
    eval_type = 'G3'
else:
    raise Exception
'''
def standardize(string):
    res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9^_]")
    string = res.sub("_", string)
    string = re.sub(r"(_)\1+","_", string).lower()
    while True:
        if len(string) == 0:
            return string
        if string[0] == "_":
            string = string[1:]
        else:
            break
    while True:
        if len(string) == 0:
            return string
        if string[-1] == "_":
            string = string[:-1]
        else:
            break
    if string[0].isdigit():
        string = "get_" + string
    return string

def change_name(name):
    change_list = ["from", "class", "return", "false", "true", "id", "and"]
    if name in change_list:
        name = "is_" + name
    return name

def dfs(node, func):
    S = set()
    if node['node_type'] == 'Action':
        name = node['description']
        flag = False
        for f in func:
            if f.endswith(name):
                S.add(f)
    for child in node['children']:
        T = dfs(child, func)
        S = S.union(T)
    return S

#fin = open('../../retrieval_unseen_hybrid2/inst/input_ours_%s_%s.json' % (eval_type, eval_id), 'r', encoding='utf-8')
fin = open('data/test_sample/G3_query_100_ground.json', 'r', encoding='utf-8')

G = json.loads(fin.read())
fin.close()
GG = {}
for x in G:
    GG[x['query_id']] = x['relevant APIs']



total_hit = 0.0
total_gold = 0.0
total_pred = 0.0
for f in os.listdir(eval_dir):
    query_id = int(f.split('_')[0])
    gold = set()
    for x in GG[query_id]:
        tool_name = x[0]
        api_name = x[1]
        final_name = '%s_for_%s' % (change_name(standardize(api_name)), standardize(tool_name))
        final_name = final_name[-64 : ]
        gold.add(final_name)
    fin = open('%s/%s' % (eval_dir, f), 'r', encoding='utf-8')
    J = json.loads(fin.read())
    fin.close()
    func = set()
    for x in J['answer_generation']['function']:
        name = x['name']
        if name != 'Finish':
            func.add(name)
    root = J['tree']['tree']
    pred = dfs(root, func)
    hit = gold.intersection(pred)
    total_hit += len(hit)
    total_gold += len(gold)
    total_pred += len(pred)

P = total_hit / total_pred
R = total_hit / total_gold
F1 = 2 * P * R / (P + R)
F2 = 5 * P * R / (4 * P + R)
print('Precision:\tRecall:\tF1:\tF2:')
print('%.4f\t%.4f\t%.4f\t%.4f' % (P, R, F1, F2))
