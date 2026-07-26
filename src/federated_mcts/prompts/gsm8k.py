# 5-shot标准prompt with格式引导
standard_prompt = """Solve the math word problem step by step and provide the final answer using the format #### [number].

Problem: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
Answer: Janet sells 16 - 3 - 4 = 9 duck eggs a day. She makes 9 * 2 = $18 every day at the farmers' market. #### 18

Problem: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
Answer: It takes 2/2 = 1 bolt of white fiber. So the total amount of fabric is 2 + 1 = 3 bolts of fabric. #### 3

Problem: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?
Answer: The cost of the house and repairs came out to 80,000 + 50,000 = $130,000. He increased the value by 80,000 * 1.5 = $120,000. So the new value is 120,000 + 80,000 = $200,000. So he made a profit of 200,000 - 130,000 = $70,000. #### 70000

Problem: James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?
Answer: He sprints 3 * 3 = 9 times. So he runs 9 * 60 = 540 meters. #### 540

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

Problem: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
Steps:
Blue fiber needed: 2 bolts
White fiber needed: half of blue fiber = 2 ÷ 2 = 1 bolt
Total fiber needed: 2 + 1 = 3 bolts
#### 3

Problem: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?
Steps:
Initial house cost: $80,000
Repair costs: $50,000
Total investment: 80,000 + 50,000 = 130,000 dollars
Value increase amount: 80,000 × 1.5 = 120,000 dollars
New house value: 80,000 + 120,000 = 200,000 dollars
Profit calculation: 200,000 - 130,000 = 70,000 dollars
#### 70000

Problem: James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?
Steps:
Sprints per session: 3
Sessions per week: 3
Total sprints per week: 3 × 3 = 9 sprints
Distance per sprint: 60 meters
Total distance per week: 9 × 60 = 540 meters
#### 540

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

Problem: A store sells apples for $2 each and oranges for $3 each. John buys 5 apples and 4 oranges. How much does he spend in total?
Current progress: John buys 5 apples at $2 each
Possible next steps:
Calculate cost of apples: 5 × 2 = 10 dollars
Calculate cost of oranges: 4 × 3 = 12 dollars  
Add total costs together

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

Problem: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and uses 4 for muffins. How much does she make selling the rest at $2 each?
Progress: Janet starts with 16 eggs
She eats 3 eggs for breakfast
Eggs after breakfast: 16 - 3 = 13 eggs
Judge: Good start with correct setup and first calculation. On track to solve the problem.
sure

Problem: A store has 24 books. If 1/3 are fiction and 1/4 are non-fiction, how many are reference books?
Progress: Fiction books: 24 × 1/3 = 8 books
Non-fiction books: 24 × 1/4 = 6 books
Total fiction and non-fiction: 8 + 6 = 14 books
Judge: Calculations are correct and logical. Ready for final step to find reference books.
sure

Problem: Tom runs 5 miles in 40 minutes. What is his speed in miles per hour?
Progress: Distance: 5 miles
Time: 40 minutes = 40/60 hours = 2/3 hours  
Speed: 5 ÷ (2/3) = 5 × 3/2 = 7.5 miles per hour
#### 7.5
Judge: Complete solution with correct unit conversion and final answer format.
sure

Problem: Lisa buys 3 notebooks at $4 each. How much does she spend?
Progress: Cost per notebook: $4
Number of notebooks: 3
Judge: Setup is correct but missing the actual calculation step.
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

Problem: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and uses 4 for muffins. She sells the rest at $2 each. How much does she make?
Solution: Janet starts with 16 eggs. She eats 3 and uses 4 for muffins. Eggs left: 16 - 3 - 4 = 9. She sells them at $2 each: 9 × 2 = 18. #### 18
Judge: Mathematical reasoning is correct, final answer is correct, and format is proper.
sure

Problem: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts total?
Solution: Blue fiber: 2 bolts. White fiber: 2 ÷ 2 = 1 bolt. Total: 2 + 1 = 3 bolts. #### 3
Judge: Correct calculation and proper format.
sure

Problem: Tom has 12 apples and gives away 1/4. How many does he have left?
Solution: Tom gives away 12 × 1/4 = 3 apples. He has 12 - 3 = 8 apples left. #### 8
Judge: Wait, let me check: 12 × 1/4 = 3, so 12 - 3 = 9, not 8. The final answer is incorrect.
impossible

Problem: Sarah earns $15 per hour and works 8 hours. How much does she earn?
Solution: Sarah works 8 hours at $15 per hour. Total earnings: 8 × 15 = 120 dollars.
Judge: Calculation is correct but missing the required #### format for final answer.
impossible

Problem: A store has 30 items. 20% are damaged. How many items are not damaged?
Solution: Damaged items: 30 × 0.2 = 6 items. Non-damaged items: 30 - 6 = 24 items. #### 24
Judge: Mathematical reasoning is sound, answer is correct, format is proper.
sure

Problem: {input}
Solution: {answer}
Judge:
"""
