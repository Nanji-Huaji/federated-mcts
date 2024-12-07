# import argparse
# from tot.methods.bfs import solve
# from tot.tasks.game24 import Game24Task

# args = argparse.Namespace(backend='lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF', temperature=0.7, task='game24', naive_run=False, prompt_sample=None, method_generate='propose', method_evaluate='value', method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)

# task = Game24Task()
# ys, infos = solve(args, task, 900)
# print(ys[0])


import re
from tot.pattern_match import check_final_result
def test_output_modfiy(idx: int, output: str):
    problem_numbers = ['1','1','4','6']
    x = problem_numbers[0] + ' ' + problem_numbers[1] + ' ' + problem_numbers[2] + ' ' + problem_numbers[3]
    split_output = output.split('\n')
    output_list = list(filter(None, split_output))
    new_output = ''
    for idx, line in enumerate(output_list):
        if(idx==0): 
            correct, cali_output = check_final_result(line, x=x)
        else:
            correct, cali_output = check_final_result(line, output_list[idx-1])
        if(correct==False):
            return {"r": 0}, output
        new_output = new_output + '\n' + cali_output
    if "(left: 24)" in output:
        return {"r": 1}, new_output
    else:
        return {"r": 0}, new_output

text = "6 / 1 = 6 (left: 1 4 6)\n6 / 1 = 6 (left: 4 6)\n4 * 6 = 24 (left: 24)\n"
a,b = test_output_modfiy(1, text)
print(a,b)