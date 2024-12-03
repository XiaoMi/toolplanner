# Copyright (C) 2024 Xiaomi Corporation.
# The source code included in this project is licensed under the Apache 2.0 license.
import json
import logging
import math
import random
import re
import requests
import sys
import time

from nltk.tokenize import word_tokenize

random.seed(42)

subset = sys.argv[1]
method = sys.argv[2]

print("subset",subset)
print("method",method)

logging.basicConfig(filename='eval_%s_%s.log' % (subset, method.split("/")[-1].split(".")[0]), level=logging.INFO)

prompt_check_solve = '''Please check whether the answer solves the query or not.
Query:
%s

Answer:
%s

If you think the answer solves the query, please answer "Yes". Otherwise, please answer "No". Please directly answer the question using the aforementioned format, and do not output any other words. Remember do not be too strict.
'''

prompt_select_best_final_answer = '''Given a user query:
%s

Please check which of the following two answers is better for the user query.
Answer (A):
%s

Answer (B):
%s

If you think the first answer is better, please answer "(A)". Otherwise, please answer "(B)". Please directly answer the question using the aforementioned format, and do not output any other words.
'''

def parse(filename):
    fin = open(filename, 'r', encoding='utf-8')
    J = json.loads(fin.read())
    fin.close()

    P = {}
    for idx in J:
        query = J[idx]['query']
        toolset = [x['name'] for x in J[idx]['available_tools'][ : -1]]
        ans = J[idx]['answer']
        total_steps = ans['total_steps']
        final_answer = ans['final_answer']
        if 'give_up_and_restart' in final_answer:
            final_answer = ''
        tool_calls = []
        detail = ans['answer_details']
        while len(detail) > 0:
            if detail[0]['role'] == 'tool':
                tool_name = re.match(r'\{\'name\': \'.*?\'', detail[0]['message'])
                tool_name = tool_name.group()[10 : -1]
                if tool_name in toolset:
                    tool_calls.append(tool_name)
            detail = detail[0]['next']
        p = {
                'query': query,
                'total_steps': total_steps,
                'succeed_tool_calling': len(tool_calls),
                'used_tool_types': len(set(tool_calls)),
                'final_answer': final_answer
        }
        P[idx] = p

    return P

ref = parse('./toolbench/tooleval/results/default_evalset/gpt-3.5-turbo_CoT/%s.json' % subset)
#hyp = parse('results/%s/%s.json' % (method, subset))
hyp = parse(method)

print("###########ref######hyp#######")
print(len(ref))
print(len(hyp))

timeout = 60
retry_interval = timeout
max_retry_interval = timeout * 2

def gen_uid():
    return hex(random.randint(0, 0xffffffff))[2 : ]



def generate(prompt, name):
    input_data = {"prompt": "你好", "history": [], "max_length": 256, "model":"gpt-4-32k-0314"}
    input_data["prompt"] = prompt
    headers = {'Content-Type': 'application/json'}
    succeed = False
    response = requests.post(url="http://", json=input_data)
    #response = response.json()
    response = json.loads(response.content)["response"]
    time.sleep(1)
    if "ERROR:Requests" in response:
        if "Please retry after" in response:
            timelen=response.split("Please retry after ")[1]
            timelen=timelen.split(" seconds.")[0]
            timelen=int(timelen)
            time.sleep(timelen+1)
            response = requests.post(url="http://", json=input_data)
            response = json.loads(response.content)["response"]


    return response

'''
def generate(prompt, name):
    data = {'query': [prompt], 'maxtokens': 256}
    headers = {'Content-Type': 'application/json'}
    succeed = False
    retry_interval = timeout
    while not succeed:
        data['uid'] = gen_uid()
        try:
            response = requests.post(url, json=data, headers=headers, timeout=timeout)
            response = response.json()
            if response['success']:
                response.update(data)
                succeed = True
            else:
                raise Exception
        except Exception as e:
            logging.info(e)
            if retry_interval <= max_retry_interval:
                logging.info('Generation Failed. Retry in %d seconds...' % retry_interval)
                time.sleep(retry_interval)
                retry_interval *= 2
            else:
                response = data.copy()
                response['name'] = name
                response['success'] = False
                logging.info('Generation Failed. Skip to next prompt.')
                return response
    response['name'] = name
    logging.info('Generation %s success.' % name)
    return response
'''
def check_solve_query(idx, i, query, answer):
    prompt = prompt_check_solve % (query, answer)
    name = 'check_solve_query_%s_%d' % (idx, i)
    response = generate(prompt, name)
    solve = None
    if len(response)>0:
        yes = False
        no = False
        tokens = word_tokenize(response)
        if 'Yes' in tokens:
            yes = True
        if 'No' in tokens:
            no = True
        if yes ^ no:
            solve = yes
    if solve is None:
        print('random choice')
        solve = bool(random.randint(0, 1))
    #response['solve'] = solve
    print("###############here response##############")
    print(response)
    return solve,response

def compare_answer_details(idx, answers):
    scores = []
    for ans in answers:
        score = 0
        score += ans['succeed_tool_calling'] * 10
        score += ans['used_tool_types'] * 5
        score -= 5 * math.log(ans['total_steps'])
        print(ans['succeed_tool_calling'], ans['used_tool_types'], ans['total_steps'])
        scores.append(score)

    highest_idx = [0]
    highest_score = scores[0]
    for i in range(1, len(scores)):
        score = scores[i]
        if score > highest_score:
            highest_idx = [i]
            highest_score = score
        elif score == highest_score:
            highest_idx.append(i)
    if len(highest_idx) > 1:
        print('random choice')
    choice = random.choice(highest_idx)
    print('idx: %s, scores: %s, choice: %d' % (idx, str(scores), choice))
    return choice
'''
def select_best_final_answer(idx, query, answers):
    prompt = prompt_select_best_final_answer % (query, answers[0], answers[1])
    name = 'select_best_final_answer_%s' % idx
    response = generate(prompt, name)
    best = None
    if response['success']:
        a = False
        b = False
        if '(A)' in response['reply']:
            a = True
        if '(B)' in response['reply']:
            b = True
        if a ^ b:
            best = int(b)
    if best is None:
        print('random choice')
        best = random.randint(0, 1)
    response['best'] = best
    print(json.dumps(response))
    return best
'''
def select_best_final_answer(idx, query, answers):
    prompt = prompt_select_best_final_answer % (query, answers[0], answers[1])
    name = 'select_best_final_answer_%s' % idx
    response = generate(prompt, name)
    best = None
    if len(response)>0:
        a = False
        b = False
        if '(A)' in response:
            a = True
        if '(B)' in response:
            b = True
        if a ^ b:
            best = int(b)
    if best is None:
        print('random choice')
        best = random.randint(0, 1)
    #response['best'] = best
    print("##########best############")
    print(best)
    return best

n_win = 0
f1=open("data/solved/"+ method.split("/")[-1],"w")
for idx in ref:
    assert(idx in hyp)
    all_empty = True
    all_nonempty = True
    is_nonempty = []
    answers = [ref[idx], hyp[idx]]
    win = 0
    for ans in answers:
        status = (ans['final_answer'] != '')
        if status:
            all_empty = False
        else:
            all_nonempty = False
        is_nonempty.append(status)

    if all_nonempty:
        all_solved = True
        all_failed = True
        is_solved = []
        for i, ans in enumerate(answers):
            status,response = check_solve_query(idx, i, ans['query'], ans['final_answer'])
            if status:
                all_failed = False
            else:
                all_solved = False
            is_solved.append(status)

            if i ==0:
                f1.write("#############################"+"\n")
                f1.write(idx+"\t"+str(i)+"\n")
                f1.write(ans['query']+"\n")
            f1.write(str(ans['final_answer'])+"\n")
            

        if all_solved:
            shortest = int(answers[0]['total_steps'])
            ans_idxs = [0]
            for i in range(1, len(answers)):
                ans = answers[i]
                if ans['total_steps'] < shortest:
                    shortest = ans['total_steps']
                    ans_idxs = [i]
                elif ans['total_steps'] == shortest:
                    ans_idxs.append(i)

            if len(ans_idxs) > 1:
                best_idx = select_best_final_answer(idx, answers[0]['query'], [answers[i]['final_answer'] for i in ans_idxs])

                win = ans_idxs[best_idx]
            else:
                win = ans_idxs[0]
            print('idx: %s, lengths: %s, win: %d' % (idx, str([ans['total_steps'] for ans in answers]), win))

        elif all_failed:
            win = compare_answer_details(idx, answers)
            print('idx: %s, win: %d' % (idx, win))
        else:
            win = random.choice([index for index, solve in enumerate(is_solved) if solve])
            print('idx: %s, win: %d' % (idx, win))
        f1.write(str(win)+"\n")
    elif all_empty:
        win = compare_answer_details(idx, answers)
        print('idx: %s, win: %d' % (idx, win))
    else:
        win = random.choice([index for index, nonempty in enumerate(is_nonempty) if nonempty])
        print('idx: %s, win: %d' % (idx, win))

    if win == 1:
        n_win += 1

print('N_WIN: %d' % n_win)
