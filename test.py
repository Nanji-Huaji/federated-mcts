# import argparse
# from tot.methods.bfs import solve
# from tot.tasks.game24 import Game24Task

# args = argparse.Namespace(backend='lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF', temperature=0.7, task='game24', naive_run=False, prompt_sample=None, method_generate='propose', method_evaluate='value', method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)

# task = Game24Task()
# ys, infos = solve(args, task, 900)
# print(ys[0])


import re
import sympy
def test_output(idx: int, output: str):
    return {'r': '(left: 24)' in output}

ys = ['8 - 1 = 7 (left: 1 3 7)\n1 + 7 = 8 (left: 3 8)\n3 * 8 = 24 (left: 24)\n', '3 * 8 = 24 (left: 1 1 24)\n1 * 24 = 24 (left: 1 24)\n24 * 1 = 24 (left: 24)\n', '3 * 8 = 24 (left: 1 1 24)\n1 * 24 = 24 (left: 1 24)\n24 / 1 = 24 (left: 24)\n', '3 * 8 = 24 (left: 1 1 24)\n1 * 24 = 24 (left: 1 24)\n1 * 24 = 24 (left: 24)\n', '\n3 * 8 = 24 (left: 1 1 24)\n1 * 1 = 1 (left: 1 24)\n1 * 24 = 24 (left: 24)\n']
a = test_output(2, '111')
print(a)
        