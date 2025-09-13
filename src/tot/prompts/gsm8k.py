# 5-shot标准prompt with格式引导
standard_prompt = """Solve the math word problem step by step and provide the final answer using the format #### [number].

Problem: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?
Answer: The cost of the house and repairs came out to 80,000 + 50,000 = $130,000. He increased the value by 80,000 * 1.5 = $120,000. So the new value is 120,000 + 80,000 = $200,000. So he made a profit of 200,000 - 130,000 = $70,000. #### 70000

Problem: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer: Weng earns $12/hour = $12/60 minutes = $0.2 per minute. She worked for 50 minutes, so she earned 50 * $0.2 = $10. #### 10

Problem: {input}
Answer: """

# 5-shot CoT prompt with格式引导
cot_prompt = """Solve the math word problem step by step. Break down the solution into clear steps and provide the final answer using the format #### [number].

FORMAT INSTRUCTIONS:
- Write each calculation step clearly
- Show your work for each computation  
- End with #### [final_number] to indicate the final answer
- Use whole numbers or decimals as appropriate

Problem: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
Steps:
Janet starts with 16 eggs per day
She eats 3 eggs for breakfast
She uses 4 eggs for muffins
Eggs left to sell: 16 - 3 - 4 = 9 eggs
She sells each egg for $2
Daily earnings: 9 × 2 = 18 dollars
#### 18

Problem: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?
Steps:
Initial house cost: $80,000
Repair costs: $50,000
Total investment: 80,000 + 50,000 = 130,000 dollars
Value increase amount: 80,000 × 1.5 = 120,000 dollars
New house value: 80,000 + 120,000 = 200,000 dollars
Profit calculation: 200,000 - 130,000 = 70,000 dollars
#### 70000

Problem: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Steps:
Hourly rate: $12 per hour
Convert to rate per minute: 12 ÷ 60 = 0.2 dollars per minute
Time worked: 50 minutes
Total earnings: 50 × 0.2 = 10 dollars
#### 10

Problem: {input}
Steps:
"""

# Propose prompt with格式引导
propose_prompt = """Given a math problem and current progress, suggest the next logical step in solving the problem.

FORMAT INSTRUCTIONS:
- Suggest one clear calculation or reasoning step
- Show the mathematical operation if applicable
- If this should be the final step, include #### [number]
- Focus on one logical progression at a time

Problem: Sarah has 24 stickers. She gives 1/3 to her sister and 1/4 to her brother. How many stickers does she have left?
Current progress: Sarah starts with 24 stickers
She gives 1/3 to her sister: 24 × 1/3 = 8 stickers
Possible next steps:
Calculate stickers given to brother: 24 × 1/4 = 6 stickers
Calculate total given away: 8 + 6 = 14 stickers
Calculate remaining: 24 - 14 = 10 stickers, then #### 10

Problem: {input}
Current progress: {current_progress}
Possible next steps:
"""

# Value prompt with格式引导
value_prompt = """Evaluate the progress towards solving this math problem. Judge if the current steps are on the right track and properly formatted (sure/likely/unlikely/impossible).

EVALUATION CRITERIA:
- Are the calculations correct?
- Is the reasoning logical?
- Are we progressing toward the final answer?
- Is the format clear and organized?
- If complete, does it end with #### [number]?

Please respond with one of: sure, likely, unlikely, impossible. Do not answer anything else. 

Problem: A store has 24 books. If 1/3 are fiction and 1/4 are non-fiction, how many are reference books?
Progress: Fiction books: 24 × 1/3 = 8 books
Non-fiction books: 24 × 1/4 = 6 books
Total fiction and non-fiction: 8 + 6 = 14 books
Judge:
sure

Problem: Lisa buys 3 notebooks at $4 each. How much does she spend?
Progress: Cost per notebook: $4
Number of notebooks: 3
Judge:
likely

Problem: {input}
Progress: {current_progress}
Judge:
"""

# Value last step prompt - 评估模型原始输出
value_last_step_prompt = """Evaluate if the given solution to this math problem is correct and properly formatted (sure/impossible).

EVALUATION CRITERIA:
- Is the mathematical reasoning correct?
- Is the final numerical answer correct?
- Is it formatted properly with #### [number]?
- Does it answer the specific question asked?

Please respond with one of: sure, impossible. Do not answer anything else.

Problem: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts total?
Solution: Blue fiber: 2 bolts. White fiber: 2 ÷ 2 = 1 bolt. Total: 2 + 1 = 3 bolts. #### 3
Judge:
sure

Problem: A store has 30 items. 20% are damaged. How many items are not damaged?
Solution: Damaged items: 30 × 0.2 = 6 items. Non-damaged items: 30 - 6 = 24 items. #### 24
Judge:
sure

Problem: {input}
Solution: {answer}
Judge:
"""
