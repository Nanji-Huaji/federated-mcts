# 5-shot standard prompt
standard_prompt = """Solve the math word problem.
Question: Mary has 5 boxes with 6 apples each. She eats 8 apples. How many apples does she have left?
Answer: 5 * 6 - 8 = 22
Question: A store sells pens for $3 each. John buys 7 pens with a $30 bill. How much change does he get?
Answer: 30 - 3 * 7 = 9
Question: Lisa walks 4 miles per hour for 3 hours, then 3 miles per hour for 2 hours. How far did she walk?
Answer: 4 * 3 + 3 * 2 = 18
Question: Tom has 48 cookies. He gives 1/4 to his sister and 1/3 of the remainder to his friend. How many cookies does he have left?
Answer: 48 - 12 - 12 = 24
Question: A farmer has 15 cows and 25 chickens. How many legs do all the animals have in total?
Answer: 15 * 4 + 25 * 2 = 110
Question: {question}
Answer:"""

# 5-shot CoT prompt with step-by-step reasoning
cot_prompt = """Solve the math word problem step by step.
Question: Mary has 5 boxes with 6 apples each. She eats 8 apples. How many apples does she have left?
Steps:
Total apples: 5 * 6 = 30
After eating: 30 - 8 = 22
#### 22

Question: A store sells pens for $3 each. John buys 7 pens with a $30 bill. How much change does he get?
Steps:
Cost of pens: 7 * 3 = 21
Change: 30 - 21 = 9
#### 9

Question: Lisa walks 4 miles per hour for 3 hours, then 3 miles per hour for 2 hours. How far did she walk?
Steps:
First part: 4 * 3 = 12 miles
Second part: 3 * 2 = 6 miles
Total: 12 + 6 = 18 miles
#### 18

Question: Tom has 48 cookies. He gives 1/4 to his sister and 1/3 of the remainder to his friend. How many cookies does he have left?
Steps:
To sister: 48 * 1/4 = 12 cookies
Remainder: 48 - 12 = 36 cookies
To friend: 36 * 1/3 = 12 cookies
Left: 36 - 12 = 24 cookies
#### 24

Question: A farmer has 15 cows and 25 chickens. How many legs do all the animals have in total?
Steps:
Cow legs: 15 * 4 = 60 legs
Chicken legs: 25 * 2 = 50 legs
Total: 60 + 50 = 110 legs
#### 110

Question: {question}
Steps:"""

# Propose next step - showing multiple possible approaches
propose_prompt = """Question: A bakery sold 45 cupcakes in the morning and 67 in the afternoon. Each cupcake costs $3. How much money did they make?
Current state: (nothing calculated yet)
Possible next steps:
Calculate morning sales: 45 * 3 = 135
Calculate afternoon sales: 67 * 3 = 201
Calculate total cupcakes: 45 + 67 = 112
Find morning revenue first: 45 cupcakes at $3 each
Find afternoon revenue first: 67 cupcakes at $3 each
Add all cupcakes then multiply: (45 + 67) * 3

Question: {question}
Current state: {state}
Possible next steps:"""

# For proposing next step when already have some calculations
propose_next_step_prompt = """Question: A bakery sold 45 cupcakes in the morning and 67 in the afternoon. Each cupcake costs $3. How much money did they make?
Previous steps:
Total cupcakes: 45 + 67 = 112
Possible next steps:
Calculate total revenue: 112 * 3 = 336
Check morning revenue: 45 * 3 = 135
Check afternoon revenue: 67 * 3 = 201
Find price per cupcake times total: 3 * 112

Question: {question}
Previous steps:
{trajectory}
Possible next steps:"""

# Value prompt for intermediate states
value_prompt = """Evaluate if the current approach will lead to the correct answer (sure/likely/unlikely/impossible)

Question: John has 20 apples and gives away 8. How many does he have?
Current: Total apples = 20
likely

Question: John has 20 apples and gives away 8. How many does he have?
Current: 20 - 8 = 12
sure

Question: John has 20 apples and gives away 8. How many does he have?
Current: 20 + 8 = 28
impossible

Question: A car travels 60 mph for 2 hours. How far did it go?
Current: Speed = 60 mph, Time = 2 hours
likely

Question: A car travels 60 mph for 2 hours. How far did it go?
Current: Distance = 60 * 2 = 120 miles
sure

Question: A car travels 60 mph for 2 hours. How far did it go?
Current: 60 / 2 = 30
impossible

Question: {question}
Current: {trajectory}
"""

# Value prompt for final answer validation
value_last_step_prompt = """Judge if the answer correctly solves the problem (sure/impossible)

Question: Mary has 10 apples and buys 5 more. How many does she have?
Answer: 10 + 5 = 15
#### 15
Judge: sure

Question: Mary has 10 apples and buys 5 more. How many does she have?
Answer: 10 - 5 = 5
#### 5
Judge: impossible

Question: A rectangle has length 8 and width 5. What is its area?
Answer: 8 * 5 = 40
#### 40
Judge: sure

Question: A rectangle has length 8 and width 5. What is its area?
Answer: 8 + 5 = 13
#### 13
Judge: impossible

Question: Tom drives 50 mph for 3 hours. How far did he travel?
Answer: 50 * 3 = 150 miles
#### 150
Judge: sure

Question: Tom drives 50 mph for 3 hours. How far did he travel?
Answer: 50 + 3 = 53 miles
#### 53
Judge: impossible

Question: {question}
Answer: {answer}
Judge:"""

# Alternative prompts for different reasoning strategies
algebraic_prompt = """Solve using algebraic thinking.
Question: {question}
Let me define variables and set up equations:"""

breakdown_prompt = """Break down the problem into smaller parts.
Question: {question}
Sub-problems:
1."""

verification_prompt = """Solve and verify the answer.
Question: {question}
Solution:"""
