# 5-shot
standard_prompt = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24.
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Input: 1 4 8 8
Answer: (8 / 4 + 1) * 8 = 24
Input: 5 5 5 9
Answer: 5 + 5 + 5 + 9 = 24
Input: {input}
'''

# 5-shot
cot_prompt = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number.
Input: 4 4 6 8
Steps:
4 + 8 = 12 (left: 4 6 12)
6 - 4 = 2 (left: 2 12)
2 * 12 = 24 (left: 24)
Answer: (6 - 4) * (4 + 8) = 24
Input: 2 9 10 12
Steps:
12 * 2 = 24 (left: 9 10 24)
10 - 9 = 1 (left: 1 24)
24 * 1 = 24 (left: 24)
Answer: (12 * 2) * (10 - 9) = 24
Input: 4 9 10 13
Steps:
13 - 10 = 3 (left: 3 4 9)
9 - 3 = 6 (left: 4 6)
4 * 6 = 24 (left: 24)
Answer: 4 * (9 - (13 - 10)) = 24
Input: 1 4 8 8
Steps:
8 / 4 = 2 (left: 1 2 8)
1 + 2 = 3 (left: 3 8)
3 * 8 = 24 (left: 24)
Answer: (1 + 8 / 4) * 8 = 24
Input: 5 5 5 9
Steps:
5 + 5 = 10 (left: 5 9 10)
10 + 5 = 15 (left: 9 15)
15 + 9 = 24 (left: 24)
Answer: ((5 + 5) + 5) + 9 = 24
Input: {input}
'''

# 1-shot
# 等待修改
propose_prompt = '''
Now we are going to use the input to play game of 24. Please imitate the example below and write out the Possible next steps based on the numbers in the Input. 
Please note: each number can only be used once, and try to avoid generating completely identical operations.
Please strictly adhere to the following output format and do not output any text other than these formats.
Please express only the latest steps when stating possible next steps, without repeating the steps that have already been expressed before.
Please do not forget that after two numbers are operated on, what remains is and only is the result of their operation.
Please note that each possible next step is independent of one another.
If the given input has already reached 24, please do not perform any operations and directly output the original input.
Input: 2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)
14 - 8 = 6 (left: 2 6 8)
14 /  2 = 7 (left: 7 8 8)
14 - 2 = 12 (left: 8 8 12)
Input: {input}
Possible next steps:
'''

propose_prompt_backup = '''
Let's play the game of 24 with the given numbers using arithmetic operations to reach 24. Follow these guidelines:
1. Use each number exactly once and create unique operations.
2. Stick to the output format provided, with no additional text.
3. List only the most recent steps, avoiding repetition.
4. Remember, after operating on two numbers, only the result of that operation remains.
5. Each next step is independent.
Input: 2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)
14 - 8 = 6 (left: 2 6 8)
14 /  2 = 7 (left: 7 8 8)
14 - 2 = 12 (left: 8 8 12)
Input: {input}
Possible next steps:
'''

# 小模型的生成能力问题不大，主要是要修改评估部分。
# 对value_prompt进行修改。

value_prompt = '''Evaluate if given numbers can reach 24 (sure/likely/impossible). Please do not restate the question, do not derive in any way other than the following example.
10 14
10 + 14 = 24
sure
11 12
11 + 12 = 23
12 - 11 = 1
11 * 12 = 132
11 / 12 = 0.91
impossible
4 4 10
4 + 4 + 10 = 8 + 10 = 18
4 * 10 - 4 = 40 - 4 = 36
(10 - 4) * 4 = 6 * 4 = 24
sure
4 9 11
9 + 11 + 4 = 20 + 4 = 24
sure
1 * 3 = 3 (left: 1 2 8)
(1 + 2) * 8 = 24 
sure
1 * 3 = 3 (left: 1 2 8)\n1 + 2 = 3 (left: 3 8)\n
3 * 8 = 24
sure
8 / 1 = 8 (left: 1 2 8)\n1 + 2 = 3 (left: 3 8)\n
3 * 8 = 24
sure
1 * 1 = 1 (left: 1 1 8)\n   1 * 8 = 8 (left: 1 2 8)\n2 / 1 = 2 (left: 6 2)\n\n\n6 * 2 = 12 (left: 4 12)\n12 - 4 = 8 (left: 4 8 12)
4 + 8 + 12 = 24
sure
1 * 1 = 1 (left: 1 1 8)\n   1 * 8 = 8 (left: 1 2 8)\n2 / 1 = 2 (left: 6 2)\n\n\n6 * 2 = 12 (left: 4 12)\n12 / 4 = 3 (left: 3 4 12)
12 + 3 * 4 = 24
sure
1 + 1 = 2 (left: 1 1 8 2)\n   8 / 1 = 8 (left: 1 2 8)\n\n1 + 2 = 3 (left: 3 8)\n3 * 8 = 24 (left: 24)\n
sure
4 * 5 = 20 (left: 6 10 20)\n20 + 10 = 30 (left: 6 30)\n30 - 6 = 24 (left: 0)
sure
5 6 6
5 + 6 + 6 = 17
(6 - 5) * 6 = 1 * 6 = 6
I cannot obtain 24 now, but numbers are within a reasonable range
likely
10 10 11
10 + 10 + 11 = 31
(11 - 10) * 10 = 10
10 10 10 are all too big
impossible
1 3 3
1 * 3 * 3 = 9
(1 + 3) * 3 = 12
1 3 3 are all too small
impossible
8 - 1 = 7 (left: 1 2 7)\n7 - 1 = 6 (left: 2 6)\n
2, 6 are too small
impossible
8 / 1 = 8 (left: 1 2 8)\n8 / 1 = 8 (left: 2 8)\n8 - 2 = 6 (left: 6, 8)
6 + 8 = 14
6 * 8 = 48
8 - 6 = 2
8 / 6 = 1.33
impossible
{input}
'''

value_last_step_prompt = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Given an input and an answer, give a judgement (sure/impossible) if the answer is correct, i.e. it uses each input exactly once and no other numbers, and reach 24. Please note: each number can only be used once.
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Judge: 
sure
Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Judge: 
sure
Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Judge: 
sure
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) + 1 = 25
Judge: 
impossible
Input: 2 9 10 12
Answer: 2 * (12 - 10) = 24
Judge: 
impossible
Input: 4 9 10 13
Answer: (13 - 4) * (10 - 9) = 24
Judge: 
impossible
Input: {input}
Answer: {answer}
Judge:'''

correction_prompt = '''
'''