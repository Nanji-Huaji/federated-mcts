# import argparse
# from tot.methods.bfs import solve
# from tot.tasks.game24 import Game24Task

# args = argparse.Namespace(backend='lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF', temperature=0.7, task='game24', naive_run=False, prompt_sample=None, method_generate='propose', method_evaluate='value', method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)

# task = Game24Task()
# ys, infos = solve(args, task, 900)
# print(ys[0])

import re
pattern = r"\(left: \d+\)"
    
# 使用re.search检查是否匹配
text = '(left: 24)'
match = re.search(pattern, text)
if match:
    print(text)

text = '(left: 24 16)'
match = re.search(pattern, text)
if match:
    print(text)

text = '(left: a)'
match = re.search(pattern, text)
if match:
    print(text)